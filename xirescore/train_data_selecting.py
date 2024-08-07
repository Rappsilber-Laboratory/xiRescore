import pandas as pd
from xirescore import readers
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_scaling import get_scaler
import logging
import numpy as np


def select(input_data, options, logger):
    """
    Select training data for Crosslink MS Machine Learning based on specified options.

    Parameters:
    - input_data: The input data for selection.
    - options: Dictionary containing various configuration options for the selection process.
    - logger: Logger instance for logging information and debugging.

    Returns:
    - A pandas DataFrame containing the selected training data.
    """

    if logger is None:
        # Set up default logging configuration if no logger is provided
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logger.getChild(__name__)

    # Extract options
    selection_mode = options['rescoring']['train_selection_mode']
    train_size_max = options['rescoring']['train_size_max']
    top_sample_size = options['rescoring']['top_sample_size']
    seed = options['rescoring']['random_seed']
    col_self_between = options['input']['columns']['self_between']
    col_fdr = options['input']['columns']['fdr']
    col_native_score = options['input']['columns']['score']
    col_target = options['input']['columns']['target']
    fdr_cutoff = options['rescoring']['train_fdr_threshold']
    val_self = options['input']['constants']['self']

    # Read input data
    df = readers.read_top_sample(input_data, logger=logger, sample=top_sample_size)
    logger.debug(f'Fetched {len(df):,.0f} top ranking samples')

    # Generate needed columns
    df = generate_columns(df, options=options, do_fdr=True, do_self_between=True)

    # Get scaler
    scaler = get_scaler(df, options, logger)

    # Selection mode: self-targets-all-decoys
    logger.info(f'Use selection mode {selection_mode}')
    if selection_mode == 'self-targets-all-decoys':
        # Create filters
        filter_self = df[col_self_between] == val_self
        filter_fdr = df[col_fdr] <= fdr_cutoff
        filter_target = df[col_target]

        # Max target size
        target_max = int(train_size_max / 2)

        # Get self targets
        train_self_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_self_targets) > target_max:
            train_self_targets = train_self_targets.sample(target_max, random_state=seed)
        logger.info(f'Taking {len(train_self_targets):,.0f} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df[filter_fdr & filter_target & ~filter_self]
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_targets):,.0f} between targets below {fdr_cutoff} FDR')

        # Get self decoy-x
        train_self_decoys = df[filter_self & ~filter_target]
        sample_min = min(
            len(train_self_decoys),
            int(train_size_max/4)
        )
        train_self_decoys = train_self_decoys.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_self_decoys):,.0f} self decoys.')

        # Get between decoy-x
        train_between_decoys = df[(~filter_self) & (~filter_target)]
        sample_min = min(
            len(train_between_decoys),
            int(train_size_max/2)-len(train_self_decoys),
        )
        train_between_decoys = train_between_decoys.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_decoys):,.0f} between decoys.')

        train_data_df = pd.concat([
            train_self_targets,
            train_between_targets,
            train_self_decoys,
            train_between_decoys
        ]).copy()

    # Selection mode: self-targets-capped-decoys
    elif selection_mode == 'self-targets-capped-decoys':
        # Create filters
        filter_self = df[col_self_between] == val_self
        filter_fdr = df[col_fdr] <= fdr_cutoff
        filter_target = df[col_target]

        # Max target size
        target_max = int(train_size_max / 2)

        # Get self targets
        train_self_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_self_targets) > target_max:
            train_self_targets = train_self_targets.sample(target_max, random_state=seed)
        logger.info(f'Taking {len(train_self_targets):,.0f} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df[filter_fdr & filter_target & ~filter_self]
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_targets):,.0f} between targets below {fdr_cutoff} FDR')

        # Get capped decoy-x
        all_taregt = df[filter_target]
        all_decoy = df[~filter_target]
        _, hist_bins = np.histogram(df[col_native_score], bins=1_000)
        hist_tt, _ = np.histogram(all_taregt[col_native_score], bins=hist_bins)
        hist_dx, _ = np.histogram(all_decoy[col_native_score], bins=hist_bins)
        hist_dx_capped = np.minimum(hist_dx, hist_tt)

        # Number of Dx to aim for
        n_decoy = min([
            hist_dx_capped.sum(),
            train_size_max/2
        ])

        # Scale decoy-x histogram
        dx_scale_fact = min(
            1,
            n_decoy / hist_dx_capped.sum(),
        )
        hist_dx_scaled = (hist_dx_capped * dx_scale_fact).astype(int)

        train_decoys = pd.DataFrame()
        for i, n in enumerate(hist_dx_scaled):
            if n == 0:
                continue
            score_min = hist_bins[i]
            score_max = hist_bins[i + 1]
            bins_samples = all_decoy[
                (all_decoy[col_native_score] >= score_min) & (all_decoy[col_native_score] < score_max)
            ]
            train_decoys = pd.concat(
                [
                    train_decoys,
                    bins_samples.sample(n=n, random_state=seed)
                ]
            )

        logger.info(f'Taking {len(train_decoys):,.0f} decoys.')

        train_data_df = pd.concat([
            train_self_targets,
            train_between_targets,
            train_decoys
        ]).copy()
    elif selection_mode == "self-target-scaled-decoy":
        # Prepare histograms
        n_bins = 1_000
        all_psms_df = df
        target_size = int(train_size_max/2)
        decoyx_size = int(train_size_max/2)
        score_col = options['input']['columns']['score']
        _, hist_bins = np.histogram(
            df[score_col],
            bins=n_bins
        )
        hist_tt, _ = np.histogram(
            df.loc[
                df['isTT'],
                score_col
            ],
            bins=hist_bins
        )
        hist_dx, _ = np.histogram(
            df.loc[
                ~df['isTT'],
                score_col
            ],
            bins=hist_bins
        )
        dx_tt_capped = np.minimum(hist_dx, hist_tt)

        # Self targets
        all_psms_self_target = df[
            df['isTT'] &
            (df["fdr_group"] == "self") &
            (df.fdr <= fdr_cutoff)
        ]
        n_self_target = min(len(all_psms_self_target), target_size)
        train_tt_df = all_psms_self_target.sample(n_self_target)
        logger.debug(f"Using {n_self_target} self-link targets FDR<={fdr_cutoff} for training")
        # Fill with between targets
        all_psms_between_target = df[
            df['isTT'] &
            (df["fdr_group"] == "between") &
            (df.fdr <= fdr_cutoff)
        ]
        n_between_target = min(target_size-n_self_target, len(all_psms_between_target))
        train_tt_df = pd.concat([
            train_tt_df,
            all_psms_between_target.sample(n_between_target)
        ])
        logger.debug(f"Filling with {n_between_target} between-link targets FDR<={fdr_cutoff} for training")

        # High scoring decoys
        n_capped_dx = dx_tt_capped.sum()
        if decoyx_size is None:
            decoyx_size = n_self_target + n_between_target

        df_decoyx = df[
            ~df['isTT']
        ]
        n_decoy = min([
            n_capped_dx,
            decoyx_size
        ])
        dx_reduce_factor = n_decoy / n_capped_dx
        hist_dx_reduced = (dx_tt_capped * dx_reduce_factor).round().astype('int')

        train_dx_df = pd.DataFrame()
        for i, n in enumerate(hist_dx_reduced):
            if n == 0:
                continue
            score_min = hist_bins[i]
            score_max = hist_bins[i+1]
            bins_samples = df_decoyx[
                (df_decoyx[score_col] >= score_min) &
                (df_decoyx[score_col] < score_max)
            ]
            train_dx_df = pd.concat([
                train_dx_df,
                bins_samples.sample(n=n)
            ])
        logger.debug(f"Using {len(train_dx_df):,.0f} decoy-x following the target-target distribution")

        train_data_df = pd.concat([
            train_dx_df,
            train_tt_df
        ])
    else:
        raise TrainDataError(f"Unknown train data selection mode: {selection_mode}.")

    return train_data_df, scaler


class TrainDataError(Exception):
    """Custom exception for train data selection errors."""
    pass

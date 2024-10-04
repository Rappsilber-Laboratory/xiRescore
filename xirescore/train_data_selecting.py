import logging

import pandas as pd
import numpy as np

from xirescore import readers
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_scaling import get_scaler


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
    top_ranking_col = options['input']['columns']['top_ranking']
    sequence_p2_col = options['input']['columns']['base_sequence_p2']
    seed = options['rescoring']['random_seed']
    col_self_between = options['input']['columns']['self_between']
    col_fdr = options['input']['columns']['fdr']
    col_native_score = options['input']['columns']['score']
    col_target = options['input']['columns']['target']
    fdr_cutoff = options['rescoring']['train_fdr_threshold']
    val_self = options['input']['constants']['self']

    # Read input data
    df = readers.read_sample(
        input_data,
        logger=logger,
        sample=top_sample_size,
        top_ranking_col=top_ranking_col,
        sequence_p2_col=sequence_p2_col,
        only_top_ranking=True,
        only_pairs=True,
    )
    logger.debug(f'Fetched {len(df)} top ranking samples')

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
        logger.info(f'Taking {len(train_self_targets)} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df[filter_fdr & filter_target & ~filter_self]
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_targets)} between targets below {fdr_cutoff} FDR')

        # Get self decoy-x
        train_self_decoys = df[filter_self & ~filter_target]
        sample_min = min(
            len(train_self_decoys),
            int(train_size_max/4)
        )
        train_self_decoys = train_self_decoys.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_self_decoys)} self decoys.')

        # Get between decoy-x
        train_between_decoys = df[(~filter_self) & (~filter_target)]
        sample_min = min(
            len(train_between_decoys),
            int(train_size_max/2)-len(train_self_decoys),
        )
        train_between_decoys = train_between_decoys.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_decoys)} between decoys.')

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
        filter_target = df[col_target].astype(bool)

        # Max target size
        target_max = int(train_size_max / 2)

        # Get self targets
        train_self_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_self_targets) > target_max:
            train_self_targets = train_self_targets.sample(target_max, random_state=seed)
        logger.info(f'Taking {len(train_self_targets)} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df[filter_fdr & filter_target & ~filter_self]
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, random_state=seed)
        logger.info(f'Taking {len(train_between_targets)} between targets below {fdr_cutoff} FDR')

        # Get capped decoy-x
        all_target = df[filter_target]
        all_decoy = df[~filter_target]
        _, hist_bins = np.histogram(df[col_native_score], bins=1_000)
        hist_tt, _ = np.histogram(all_target[col_native_score], bins=hist_bins)
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
        hist_dx_scaled = (hist_dx_capped * dx_scale_fact).round().astype(int)

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

        logger.info(f'Taking {len(train_decoys)} decoys.')

        train_data_df = pd.concat([
            train_self_targets,
            train_between_targets,
            train_decoys]
        ).copy()
    else:
        raise TrainDataError(f"Unknown train data selection mode: {selection_mode}.")

    return train_data_df, scaler


class TrainDataError(Exception):
    """Custom exception for train data selection errors."""
    pass

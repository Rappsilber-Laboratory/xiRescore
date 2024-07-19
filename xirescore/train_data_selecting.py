import pandas as pd
import readers
from bi_fdr import self_or_between_mp, calculate_bi_fdr
import logging
import numpy as np

def select(input_data, options, logger):
    if logger is None:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logger.getChild(__name__)

    # Get all options
    selection_mode = options['rescoring']['train_selection_mode']
    train_size_max = options['rescoring']['train_size_max']
    seed = options['rescoring']['random_seed']
    col_self_between = options['input']['columns']['self_between']
    col_fdr = options['input']['columns']['fdr']
    col_native_score = options['input']['columns']['native_score']
    col_target = options['input']['columns']['target']
    fdr_cutoff = options['rescoring']['train_fdr_threshold']
    val_self = options['input']['constants']['self']
    col_prot1 = options['input']['columns']['protein_p1']
    col_prot2 = options['input']['columns']['protein_p2']

    df = readers.read_top_sample(input_data)
    if 'fdr_group' not in options['input']['columns']:
        df['fdr_group'] = self_or_between_mp(
            df,
            col_prot1=col_prot1,
            col_prot2=col_prot2,
        )
    if 'fdr' not in options['input']['columns']:
        df['fdr'] = calculate_bi_fdr(
            df,
            score_col=col_native_score,
            fdr_group_col=getattr(options['input']['columns'], 'fdr_group', 'fdr_group'),
            decoy_class=options['input']['columns']['decoy_class'],
        )

    if selection_mode == 'self-targets-all-decoys':
        logger.info(f'Use selection mode {selection_mode}')
        # Create filters
        filter_self = df[col_self_between] == val_self
        filter_fdr = df[col_fdr] <= fdr_cutoff
        filter_target = df[col_target]

        # Max target size
        target_max = int(train_size_max/2)

        # Get self targets
        train_self_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_self_targets) > target_max:
            train_self_targets.sample(target_max, random_state=seed)
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
        logger.info(f'Taking {len(train_self_decoys)} between decoys.')

        return pd.concat([
            train_self_targets,
            train_between_targets,
            train_self_decoys,
            train_between_decoys
        ]).copy()

    if selection_mode == 'self-targets-capped-decoys':
        logger.info(f'Use selection mode {selection_mode}')
        # Create filters
        filter_self = df[col_self_between] == val_self
        filter_fdr = df[col_fdr] <= fdr_cutoff
        filter_target = df[col_target]

        # Max target size
        target_max = int(train_size_max/2)

        # Get self targets
        train_self_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_self_targets) > target_max:
            train_self_targets.sample(target_max, random_state=seed)
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
        all_taregt = df[filter_target]
        all_decoy = df[~filter_target]
        _, hist_bins = np.histogram(df[col_native_score],   bins=1_000)
        hist_tt, _ = np.histogram(all_taregt[col_native_score], bins=hist_bins)
        hist_dx, _ = np.histogram(all_decoy[col_native_score], bins=hist_bins)
        hist_dx_capped = np.minimum(hist_dx, hist_tt)

        # Number of Dx to aim for
        n_decoy = min([
            hist_dx_capped.sum(),
            train_size_max/2
        ])

        # Scale decoy-x histogram
        dx_scale_fact = max(
            1,
            n_decoy / hist_dx_capped.sum(),
        )
        hist_dx_scaled = (hist_dx_capped * dx_scale_fact).astype(int)

        train_decoys = pd.Index([])
        for i, n in enumerate(hist_dx_scaled):
            if n == 0:
                continue
            score_min = hist_bins[i]
            score_max = hist_bins[i+1]
            bins_samples = all_decoy[
                (all_decoy[col_native_score] >= score_min) &
                (all_decoy[col_native_score] < score_max)
            ]
            train_decoys = pd.concat([
                train_decoys,
                bins_samples.sample(n=n, random_state=seed)
            ])

        logger.info(f'Taking {len(train_decoys)} decoys.')

        return pd.concat([
            train_self_targets,
            train_between_targets,
            train_decoys,
        ]).copy()
    else:
        raise TrainDataError(f"Unknown train data selection mode: {selection_mode}.")


class TrainDataError(Exception):
    pass

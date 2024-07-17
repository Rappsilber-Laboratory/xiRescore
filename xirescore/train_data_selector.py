import pandas as pd

def select(df, options):
    # Get all options
    selection_mode = options['rescoring']['train_selection_mode']
    train_size_max = options['rescoring']['train_size_max']
    seed = options['rescoring']['random_seed']
    col_self_between = options['input']['columns']['self_between']
    col_fdr = options['input']['columns']['fdr']
    col_target = options['input']['columns']['target']
    fdr_cutoff = options['rescoring']['train_fdr_threshold']
    val_self = options['input']['constants']['self']

    if selection_mode == 'target_follow_capped':
        # Create filters
        filter_self = df[col_self_between] == val_self
        filter_fdr = df[col_fdr] <= fdr_cutoff
        filter_target = df[col_target]

        # Max target size
        target_max = int(train_size_max/2)

        # Get self targets
        train_targets = df[filter_fdr & filter_target & filter_self]
        if len(train_targets) > target_max:
            train_targets.sample(target_max, random_state=seed)

        # Get between targets
        if len(train_targets) > target_max:
            train_between_targets = df[filter_fdr & filter_target & ~filter_self]
            if len(train_between_targets) > target_max - len(train_targets):
                train_between_targets = train_between_targets.sample(target_max - len(train_targets), random_state=seed)
            train_targets = pd.concat([
                train_targets,
                train_between_targets,
            ])

        # Get self decoy-x
        # TODO
    else:
        raise TrainDataError(f"Unknown train data selection mode: {selection_mode}.")

class TrainDataError(Exception):
    pass

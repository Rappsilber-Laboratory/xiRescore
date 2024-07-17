import numpy as np

default_options = {
    'input': {
        'columns': {
            'score': 'match_score',
            'fdr': 'fdr',
            'base_sequence_p1': 'base_sequence_p1',
            'base_sequence_p2': 'base_sequence_p2',
            'train_flag': None,
            'target': 'isTT',
            'self_between': 'fdr_group',
            'feature_prefix': 'feat_',
            'features': []
        },
        'constants': {
            'self': 'self',
            'between': 'between',
        },
    },
    'rescoring': {
        'train_fdr_threshold': 0.01,  # Threshold for trian mode 'fdr'
        'train_sample_mode': 'target_follow_capped',
        'self_between_balanced': True,
        'train_size_max': 10_000,
        'model_class': 'linear_model',
        'model_name': 'LinearRegression',
        'minimize_metric': True,
        'metric_name': 'f1_score',
        'model_params': {
            "C": np.logspace(-3, 2, 6),
            "solver": ["liblinear"],
            "penalty": ["l1", "l2"],
            "random_state": [42],
            "class_weight": ["balanced", None, {0: 2, 1: 1}]
        },
        'n_splits': 5,
        'max_jobs': -1,
        'random_seed': 0,
    },
    'output': {
        'columns': {
            'rescore': 'rescore',
            'fold': 'fold',
        }
    },
}

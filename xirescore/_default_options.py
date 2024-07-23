import numpy as np

default_options = {
    'input': {
        'columns': {
            'score': 'match_score',
            'base_sequence_p1': 'base_sequence_p1',
            'base_sequence_p2': 'base_sequence_p2',
            'protein_p1': 'protein_p1',
            'protein_p2': 'protein_p2',
            'train_flag': None,
            'target': 'isTT',
            'native_score': 'match_score',
            'feature_prefix': 'feat_',
            'features': [],
            'csm_id': None,
            'spectrum_id': None,
            'decoy_class': 'decoy_class',
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
        'train_size_max': 20_000,
        'model_class': 'linear_model',
        'model_name': 'LogisticRegression',
        'minimize_metric': True,
        'metric_name': 'f1_score',
        'model_params': {
            "C": np.logspace(-3, 2, 6),
            "solver": ["liblinear"],
            "penalty": ["l1", "l2"],
            "class_weight": ["balanced", None, {0: 2, 1: 1}]
        },
        'n_splits': 5,
        'max_jobs': -1,
        'random_seed': 0,
        'logit_result': True,
    },
    'output': {
        'columns': {
            'rescore': 'rescore',
            'fold': 'fold',
        }
    },
}

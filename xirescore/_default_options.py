import numpy as np

default_options = {
    'input': {
        'columns': {
            # Boolean column indicating if a match has
            # the highest score for the specific spectrum.
            # If you don't allow for multiple matches for one spectrum
            # this column would only contain ``True`` values.
            'top_ranking': 'top_ranking',

            # The main score to use for FDR calculation and therefore training data selection.
            'score': 'feature_match_score',

            # The AA sequence of the first peptide without any modifications.
            'base_sequence_p1': 'base_sequence_p1',

            # The AA sequence of the second peptide without any modifications.
            'base_sequence_p2': 'base_sequence_p2',

            # Semicolon separated list of proteins associated with first peptide.
            'protein_p1': 'protein_p1',

            # Semicolon separated list of proteins associated with second peptide.
            'protein_p2': 'protein_p2',

            # Boolean column indicating if the first peptide is a decoy.
            'decoy_p1': 'is_decoy_p1',

            # Boolean column indicating if the second peptide is a decoy.
            'decoy_p2': 'is_decoy_p2',

            # FDR calculated for the top ranking matches.
            # Will be generated if not present in input.
            'fdr': 'fdr',

            # Information if a CSM is a self (homomeric) or between (heteromeric) match.
            # Specific classification values are defined under input.constants.
            'self_between': 'fdr_group',

            # Boolean column
            'target': 'isTT',

            # Decoy class information. Indicates if the CSM is a target-target ("TT"),
            # target-decoy ("TD") or decoy-decoy ("DD")
            'decoy_class': 'decoy_class',

            # Feature columns prefix. All columns starting with this prefix
            # will be considered feature columns.
            'feature_prefix': 'feature_',

            # Additional columns to be considered features
            'features': [],

            # Columns that uniquely identify a CSM. If None all columns of the input will be used.
            'csm_id': None,

            # Columns that uniquely identify a spectrum. Will be used for new top_ranking calculation.
            'spectrum_id': ['spectrum_id'],
        },
        'constants': {
            # Self classification value for column self_between.
            'self': 'self',

            # Between classification value for column self_between.
            'between': 'between',

            # String added to decoy protein names. Will be removed for self/between classification.
            'decoy_adjunct': 'REV_',
        },
    },
    'rescoring': {
        # FDR threshold for training samples.
        'train_fdr_threshold': 0.01,

        # Method on how to sample the training data.
        'train_selection_mode': 'self-targets-all-decoys',

        # Number of samples used for training data selection.
        'top_sample_size': 1_000_000,

        # Maximum number of training samples
        'train_size_max': 20_000,

        # Number of spectra that will be loaded and rescored at once.
        'spectra_batch_size': 100_000,

        # SKLearn module of the desired classifier.
        'model_class': 'linear_model',

        # Class name of the desired classifier.
        'model_name': 'LogisticRegression',

        # Name of the desired metric to evaluate models from sklearn.metrics.
        'metric_name': 'log_loss',

        # Boolean indicating if the metric is better when smaller.
        'minimize_metric': True,

        # Model parameters for hyperparameter optimization.
        'model_params': {
            "C": np.logspace(-3, 2, 6),
            "solver": ["liblinear"],
            "penalty": ["l1", "l2"],
            "class_weight": ["balanced", None, {0: 2, 1: 1}]
        },

        # Number of k-fold splits.
        'n_splits': 5,

        # Limit of how many CPUs to use.
        'max_jobs': -1,

        # Random seed for Python random module and numpy.
        'random_seed': 0,

        # Boolean indicating if logit should be applied to the rescoring result.
        # This can make the rescore more normally distributed with certain
        # classifiers. (E.g. LinearRegression or SVC)
        'logit_result': True,

        # Scaler to use for the features
        'scaler': 'StandardScaler',

        # Scaler's keyword arguments
        'scaler_params': {}
    },
    'output': {
        'columns': {
            # Name of the column containing the new score.
            # Also base name for each model's separate prediction
            # and training slice information.
            'rescore': 'rescore',
        }
    },
}

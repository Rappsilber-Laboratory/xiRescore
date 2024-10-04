from typing import Callable
from functools import partial
import logging
import importlib
import multiprocess as mp

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

from xirescore import async_result_resolving
from xirescore.NoOverlapKFold import NoOverlapKFold


def get_hyperparameters(train_df, cols_features, splits, options,
                        logger: logging.Logger = None, loglevel=logging.DEBUG):
    # Create new logger or create child logger from existing one
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(loglevel)
    else:
        logger = logger.getChild(__name__)

    # Get peptide sequence columns
    cols_pepseq = [
        options['input']['columns']['base_sequence_p1'],
        options['input']['columns']['base_sequence_p2'],
    ]

    # Get column for target labeling
    col_label = options['input']['columns']['target']

    # Get DataFrames for peptide sequences, features and labels
    pepseq_df = train_df[cols_pepseq]
    features_df = train_df[cols_features]
    labels_df = train_df[col_label].astype(bool)

    # Import model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model_module = importlib.import_module(f"sklearn.{model_class}")
    model: ClassifierMixin.__class__ = getattr(model_module, model_name)

    # Check for single core model
    is_mp_model = hasattr(model, 'n_jobs')

    # Get random seed and number of k-fold splits
    seed = options['rescoring']['random_seed']
    n_splits = options['rescoring']['n_splits']

    # Get the list of hyperparameter configurations
    hyperparam_grid = list(ParameterGrid(options['rescoring']['model_params']))

    if splits is None:
        # Get non peptide overlapping k-fold splits
        kf = NoOverlapKFold(
            n_splits,
            random_state=seed,
            shuffle=True,
            pep1_id_col=options['input']['columns']['base_sequence_p1'],
            pep2_id_col=options['input']['columns']['base_sequence_p2'],
            target_col=col_label,
            logger=logger,
        )
        splits = kf.splits_by_peptides(
            df=train_df,
            pepseqs=pepseq_df
        )

    max_jobs = options['rescoring']['max_jobs']
    if max_jobs < 1:
        max_jobs = mp.cpu_count()-1

    logger.info(f'Using {max_jobs} CPU cores')

    # Create partial function for parameter testing
    logger.debug(f"Metric name: {options['rescoring']['metric_name']}")
    param_try_job = partial(
        _try_parameters,
        features_df=features_df,
        labels_df=labels_df,
        splits=splits,
        options=options,
        logger=logger
    )

    with mp.Pool(processes=max_jobs) as pool:
        # Only run multiprocessing for single core models
        if is_mp_model:
            param_scores = [
                param_try_job(params=params)
                for params in hyperparam_grid
            ]
        else:
            param_scores = [
                pool.apply_async(
                    param_try_job,
                    kwds=dict(params=params)
                )
                for params in hyperparam_grid
            ]

        # Resolve (potentially) async results
        param_scores = async_result_resolving.resolve(param_scores, logger=logger)

    # Get the best parameters based on metric
    if options['rescoring']['minimize_metric']:
        best_params_i = np.argmin(param_scores)
    else:
        best_params_i = np.argmax(param_scores)

    best_params = hyperparam_grid[best_params_i]
    logger.info(f"Best hyperparameter: {best_params}")

    return best_params


def _try_parameters(features_df, labels_df, splits, params, options, logger: logging.Logger):
    # Create child logger for parameter configuration
    #logger = logger.getChild(f"{hex(abs(hash(str(params))))}")
    logger.debug(f"Params: {params}")

    # Import classifier model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model_module = importlib.import_module(f"sklearn.{model_class}")
    model: ClassifierMixin.__class__ = getattr(model_module, model_name)

    # Import metric function
    metric_name = options['rescoring']['metric_name']
    logger.debug(f"Metric name: {options['rescoring']['metric_name']}")
    metric: Callable = getattr(sklearn.metrics, metric_name)

    # Initialize result lists
    scores = []
    accuracies = []
    balanced_accuracies = []
    for cvi, fold in enumerate(splits):
        logger.debug(f"Fold {cvi}")

        # Unpack fold
        train_idx, test_idx = fold

        # Get fold's train features and labels
        fold_train_features_df = features_df.loc[train_idx]
        fold_train_labels_df = labels_df.loc[train_idx]

        # Get fold's test features and labels
        fold_test_features_df = features_df.loc[test_idx]
        fold_test_labels_df = labels_df.loc[test_idx]

        # Train fold classifier
        clf = model(**params)
        clf.fit(fold_train_features_df, fold_train_labels_df)
        test_predictions = clf.predict(fold_test_features_df)

        # Evaluate fold model
        score = metric(fold_test_labels_df, test_predictions, labels=[0, 1])
        accuracy = accuracy_score(fold_test_labels_df, test_predictions,)
        balanced_accuracy = balanced_accuracy_score(fold_test_labels_df, test_predictions,)

        # Store evaluation
        scores.append(score)
        accuracies.append(accuracy)
        balanced_accuracies.append(balanced_accuracy)

    # Log fold performances
    logger.debug(f"Scores: {scores}")
    logger.debug(f"Accuracies: {accuracies}")
    logger.debug(f"Balanced accuracies: {balanced_accuracies}")

    # Return mean score
    return np.mean(scores)

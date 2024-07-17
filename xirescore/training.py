from typing import Callable

from sklearn.model_selection import ParameterGrid
import importlib
import multiprocess as mp
import NoOverlapKFold
import logging
import numpy as np
from sklearn.base import ClassifierMixin
from functools import partial
import async_result_resolver
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def train(train_df, cols_features, clf_params, options,
          logger: logging.Logger = None, loglevel=logging.DEBUG):
    # Create new logger or create child logger from existing one
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(loglevel)
    else:
        logger = logger.getChild(__name__)

    # Get peptide sequence columns
    cols_pepseq = [
        options['input']['base_sequence_p1'],
        options['input']['base_sequence_p2'],
    ]

    # Get column for target labeling
    col_label = options['input']['target']

    # Get DataFrames for peptide sequences, features and labels
    pepseq_df = train_df[cols_pepseq]
    features_df = train_df[cols_features]
    labels_df = train_df[col_label]

    # Import model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model = importlib.import_module(f"sklearn.{model_class}.{model_name}")

    # Check for single core model
    is_mp_model = hasattr(model, 'n_jobs')

    # Get random seed and number of k-fold splits
    seed = options['rescoring']['random_seed']
    n_splits = options['rescoring']['n_splits']

    # Get non peptide overlapping k-fold splits
    kf = NoOverlapKFold(n_splits, random_state=seed, shuffle=True, logger=logger)
    splits = kf.splits_by_peptides(train_df, pepseq_df)

    max_jobs = min([
        options['rescoring']['max_jobs'],
        mp.cpu_count() - 1,
    ])

    fold_clfs = []
    with mp.Pool(processes=max_jobs) as pool:
        for cvi, fold in enumerate(splits):
            # Only run multiprocessing for single core models
            if is_mp_model:
                fold_clf = [
                    lambda: train_fold(
                        features_df=features_df,
                        labels_df=labels_df,
                        fold=fold,
                        params=clf_params,
                        options=options,
                        logger=logger.getChild(f"fold{cvi}")
                    )
                ]
            else:
                fold_clf = pool.async_apply(
                    train_fold,
                    kwds=dict(
                        features_df=features_df,
                        labels_df=labels_df,
                        fold=fold,
                        params=clf_params,
                        options=options,
                        logger=logger.getChild(f"fold{cvi}")
                    )
                )
            # Add job to the
            fold_clfs.append(fold_clf)
        # Resolve (potentially) async results
        clfs = async_result_resolver.resolve(fold_clfs, logger=logger)
    return clfs


def train_fold(features_df, labels_df, fold, params, options, logger: logging.Logger):
    # Import classifier model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model: ClassifierMixin = importlib.import_module(f"sklearn.{model_class}.{model_name}")

    # Import metric function
    metric_name = options['rescoring']['metric_name']
    metric: Callable = importlib.import_module(f"sklearn.metrics.{metric_name}")

    # Unpack fold
    train_idx, test_idx = fold

    # Get fold's train features and labels
    fold_train_features_df = features_df[train_idx]
    fold_train_labels_df = labels_df[train_idx]

    # Get fold's test features and labels
    fold_test_features_df = features_df[test_idx]
    fold_test_labels_df = labels_df[test_idx]

    # Train fold classifier
    clf = model(**params)
    clf.fit(fold_train_features_df, fold_train_labels_df)
    test_predictions = clf.predict(fold_test_features_df)

    # Evaluate fold model
    score = metric(fold_test_labels_df, test_predictions)
    accuracy = accuracy_score(fold_test_labels_df, test_predictions, )
    balanced_accuracy = balanced_accuracy_score(fold_test_labels_df, test_predictions, )
    logger.info(f"Score: {score}")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Balanced accuracy: {balanced_accuracy}")

    return clf

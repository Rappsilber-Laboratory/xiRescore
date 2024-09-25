from logging import Logger

import numpy as np
import pandas as pd


def get_features(df: pd.DataFrame, options: dict, logger: Logger):
    features_const = options['input']['columns']['features']
    feat_prefix = options['input']['columns']['feature_prefix']
    features_prefixes = [
        c for c in df.columns if str(c).startswith(feat_prefix)
    ]
    features = features_const + features_prefixes

    absent_features = [
        f
        for f in features
        if f not in df.columns
    ]
    nan_features = [
        f
        for f in features
        if (f not in absent_features) and any(pd.isna(df[f].values))
    ]
    features = [
        f
        for f in features
        if f not in (absent_features + nan_features)
    ]

    if len(nan_features) > 0:
        logger.warning(f"Dropped features with NaN values: {nan_features}")
    if len(absent_features) > 0:
        logger.warning(f"Did not find some features: {absent_features}")

    return features

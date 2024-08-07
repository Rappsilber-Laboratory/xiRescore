from sklearn.preprocessing import StandardScaler
import pandas as pd
from xirescore.feature_extracting import get_features
from logging import Logger


def get_scaler(df: pd.DataFrame, options: dict, logger: Logger):
    """
    Normalize the features and drop NaN-values if necessary.
    """
    features = get_features(df, options, logger)
    df_features = df[features]

    std_scaler = StandardScaler()
    std_scaler.fit(df_features)

    return std_scaler

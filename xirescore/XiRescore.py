"""Main module."""

from _default_options import default_options
import input_validator
import train_data_selector
from hyperparameter_optimizer import get_hyperparameters

import pandas as pd
import deepmerge
import logging
from sklearn.preprocessing import StandardScaler


class XiRescore:
    _options = default_options
    _models = []
    _logger = logging.getLogger(__name__)
    _feat_cols = []

    def __init__(self, input_df, options=dict(), logger=None, loglevel=logging.DEBUG):
        # Store raw DataFrame
        self._df = input_df

        # Apply override default options with user-supplied options
        self._options = deepmerge.merge_or_raise(
            self._options,
            options
        )

        # Use supplied logger if present
        if logger is not None:
            self._logger = logger
        self._loglevel = loglevel

        # Get all columns that begin with the feature prefix
        self._feat_cols = self._options['input']['columns']['features']
        self._feat_cols.append(
            [
                c for c in self._df.columns
                if c.beginswith(self._options['input']['columns']['feature_prefix'])
            ]
        )

        # Validate that the input DataFrame contains necessary columns
        self._validate_input()

        # Log a summary of the input data
        self._input_summary()

    def run(self) -> pd.DataFrame:
        self._choose_train_data()
        self._hyperparam_opt()
        self._train()
        self._rescore_train()
        self._rescore_rest()
        self._rescore_summary()

    def _normalize_and_cleanup(self):
        """
        Normalize the features and drop NaN-values if necessary.
        """
        features = self._df[self._feat_cols]

        std_scaler = StandardScaler()
        std_scaler.fit(features)
        features_scaled = pd.DataFrame(std_scaler.transform(features))

        self._df_scaled = self._df.copy()
        self._df_scaled[features_scaled.columns] = features_scaled

    def _choose_train_data(self):
        self._train_df = train_data_selector(self._df, self._options)

    def _hyperparam_opt(self):
        self._model_params = get_hyperparameters(
            train_df=self._train_df,
            cols_features=self._feat_cols,
            options=self._options,
            logger=self._logger,
            loglevel=self._loglevel,
        )

    def _input_summary(self):
        pass  # TODO

    def _validate_input(self):
        input_validator.validate(self._df, self._options, self._logger)

    def _train(self):
        pass  # TODO

    def _rescore_train(self):
        pass  # TODO

    def _rescore_rest(self):
        pass  # TODO

    def _rescore_summary(self):
        pass  # TODO

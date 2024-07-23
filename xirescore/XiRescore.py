"""Main module."""
from _default_options import default_options
import train_data_selecting
import training
import readers
import writers
import rescoring
from hyperparameter_optimizing import get_hyperparameters

import pandas as pd
import numpy as np
import deepmerge
import logging
import random
from math import ceil
from sklearn.preprocessing import StandardScaler


class XiRescore:
    _options = default_options
    _logger = logging.getLogger(__name__)
    train_df = None
    splits = None
    models = []

    def __init__(self,
                 input_path,
                 output_path=None,
                 options=dict(),
                 cols_csm_id=None,
                 cols_spectrum_id=None,
                 logger=None,
                 loglevel=logging.DEBUG):
        # Set random seed
        seed = options['rescoring']['random_seed']
        np.random.seed(seed)
        random.seed(seed)

        # Store input data path
        self._input = input_path
        if output_path is None:
            # Store output in new DataFrame if no path is given
            self._output = pd.DataFrame()
        else:
            self._output = output_path

        # Apply override default options with user-supplied options
        self._options = deepmerge.merge_or_raise(
            self._options,
            options
        )

        # Store CSM ID columns
        self._cols_csm_id = cols_csm_id

        # Store Spectrum ID columns
        self._cols_spectrum = cols_spectrum_id

        # Use supplied logger if present
        if logger is not None:
            self._logger = logger
        self._loglevel = loglevel

    def run(self):
        self.train()
        self.rescore()

    def train(self, train_df=None):
        if train_df is None:
            train_df = train_data_selecting.select(
                self._input,
                self._options,
                self._logger
            )
        self.train_df = self._normalize_and_cleanup(train_df)
        cols_features = self._get_features(self.train_df)

        model_params = get_hyperparameters(
            train_df=self.train_df,
            cols_features=cols_features,
            options=self._options,
            logger=self._logger,
            loglevel=self._loglevel,
        )

        self.models, self.splits = training.train(
            train_df=self.train_df,
            cols_features=self._get_features(self.train_df),
            clf_params=model_params,
            logger=self._logger,
            options=self._options,
        )

    def _normalize_and_cleanup(self, df):
        """
        Normalize the features and drop NaN-values if necessary.
        """
        features = self._get_features(df)
        df_features = df[features]
        n_all_cols = len(df_features.columns)
        df_features = df_features.dropna(axis='columns')
        n_dropped_cols = n_all_cols-len(df_features.columns)
        self._logger.info(f"Dropped {n_dropped_cols} columns with NaN-values of {n_all_cols}.")

        std_scaler = StandardScaler()
        std_scaler.fit(df_features)
        df_features_scaled = pd.DataFrame(std_scaler.transform(df_features))

        df.loc[:, df_features_scaled.columns] = df_features_scaled

        return df

    def _get_features(self, df):
        features_const = self._options['input']['columns']['features']
        feat_prefix = self._options['input']['columns']['feature_prefix']
        features_prefixes = [
            c for c in df.columns if c.beginswith(feat_prefix)
        ]
        return features_const + features_prefixes

    def rescore(self, df=None, spectra_batch_size=100_000):
        cols_spectra = self._options['input']['columns']['spectrum']
        col_rescore = self._options['output']['columns']['rescore']
        if self._options['input']['columns']['csm_id'] is None:
            col_csm = list(self.train_df.columns)
        else:
            col_csm = self._options['input']['columns']['csm_id']

        apply_logit = self._options['rescoring']['logit_result']
        if df is None:
            input_data = self._input
        else:
            input_data = df.copy()

        # Read spectra list
        spectra = readers.read_spectra_ids(input_data, cols_spectra)

        # Sort spectra
        spectra = spectra.sort_values(by=list(spectra.columns))

        # Calculate number of batches
        n_batches = ceil(len(spectra)/spectra_batch_size)

        # Iterate over spectra batches
        df_rescored = pd.DataFrame()
        for i_batch in range(n_batches):
            # Define batch borders
            spectra_range = spectra.iloc[
                i_batch*spectra_batch_size:(i_batch+1)*spectra_batch_size
            ]
            spectra_from = spectra_range.iloc[0]
            spectra_to = spectra_range.iloc[-1]

            # Read batch
            df_batch = readers.read_spectra_range(
                input=input_data,
                spectra_from=spectra_from,
                spectra_to=spectra_to,
                spectra_cols=cols_spectra
            )

            # Rescore batch
            df_batch = df_batch.merge(
                rescoring.rescore(
                    self.models,
                    df=df_batch,
                    rescore_col=col_rescore,
                    apply_logit=apply_logit
                ),
                left_index=True,
                right_index=True,
                validate='1:1',
            )

            # Rescore training data only with test fold classifier
            df_slice = self.train_df.loc[:, col_csm]
            df_slice[f'{col_rescore}_slice'] = -1
            for i, (_, idx_test) in enumerate(self.splits):
                df_slice.loc[idx_test, f'{col_rescore}_slice'] = i

            df_batch = df_batch.merge(df_slice)

            df_batch.loc[df_batch[f'{col_rescore}_slice'] > -1, col_rescore] = df_batch.loc[
                df_batch[f'{col_rescore}_slice'] > -1
            ].apply(
                _select_right_score,
                col_rescore=col_rescore,
                axis=1,
            )

            # Calculate top_ranking
            df_batch_top_rank = df_batch.groupby(self._cols_spectrum).agg({
                f'rescore_max': pd.NamedAgg(column='rescore', aggfunc='max')
            })
            df_batch.merge(
                df_batch_top_rank,
                left_on=list(self._cols_spectrum),
                right_index=True
            )
            df_batch['top_ranking'] = df_batch['rescore'] == df_batch[f'rescore_max']

            # Store collected matches
            if type(self._output) is pd.DataFrame:
                df_rescored = pd.DataFrame([
                    df_rescored,
                    df_batch
                ])
            else:
                writers.append_rescorings(self._output, df_batch)

        # Keep rescored matches when no output is defined
        if type(self._output) is pd.DataFrame:
            self._output = pd.concat([
                self._output,
                df_rescored
            ])

    def get_rescored_output(self):
        if type(self._output) is pd.DataFrame:
            return self._output
        else:
            raise XiRescoreError('Not available for file or DB output.')


def _select_right_score(row, col_rescore):
    n_slice = row[f"{col_rescore}_slice"]
    return row[f'{col_rescore}_{n_slice}']


class XiRescoreError(Exception):
    """Custom exception for train data selection errors."""
    pass

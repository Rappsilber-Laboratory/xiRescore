"""Main module."""
from xirescore._default_options import default_options
from xirescore import train_data_selecting
from xirescore import training
from xirescore import readers
from xirescore import writers
from xirescore import rescoring
from xirescore.hyperparameter_optimizing import get_hyperparameters

import pandas as pd
import numpy as np
from deepmerge import Merger
import logging
import random
from math import ceil
from sklearn.preprocessing import StandardScaler
from collections.abc import Collection

options_merger = Merger(
    # pass in a list of tuple, with the
    # strategies you are looking to apply
    # to each type.
    [
        (dict, ["merge"]),
        (list, ["override"]),
        (set, ["override"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"]
)


class XiRescore:
    _options = default_options
    _logger = logging.getLogger(__name__)
    _true_random_ctr = 0
    train_df: pd.DataFrame
    """
    Data used for training the models. Kept to not rescore training samples with models they have been trained on.
    """
    splits: Collection[tuple[pd.Index, pd.Index]]
    """
    K-fold splits of model training. Kept to not rescore training samples with models they have been trained on.
    """
    models = []

    def __init__(self,
                 input_path,
                 output_path=None,
                 options=dict(),
                 logger=None,
                 loglevel=logging.DEBUG):
        # Apply override default options with user-supplied options
        self._options = options_merger.merge(
            self._options,
            options
        )

        # Set random seed
        seed = self._options['rescoring']['random_seed']
        self._true_random_seed = random.randint(0, 2**32-1)
        np.random.seed(seed)
        random.seed(seed)

        # Store input data path
        self._input = input_path
        if output_path is None:
            # Store output in new DataFrame if no path is given
            self._output = pd.DataFrame()
        else:
            self._output = output_path

        # Use supplied logger if present
        if logger is not None:
            self._logger = logger
        self._loglevel = loglevel

    def run(self):
        """
        Run training and rescoring of the input data and write to output
        """
        self._logger.info("Start full train and rescore run")
        self.train()
        self.rescore()

    def train(self, train_df: pd.DataFrame = None):
        """
        Run training on input data or on the passed DataFrame if provided.

        Parameters:
        - train_df: Data to be used training instead of input data.
        """
        self._logger.info('Start training')
        if train_df is None:
            train_df = train_data_selecting.select(
                self._input,
                self._options,
                self._logger
            )
        self.train_df = self._normalize_and_cleanup(train_df)
        cols_features = self._get_features()

        self._logger.info("Perform hyperparameter optimization")
        model_params = get_hyperparameters(
            train_df=self.train_df,
            cols_features=cols_features,
            options=self._options,
            logger=self._logger,
            loglevel=self._loglevel,
        )

        self._logger.info("Train models")
        self.models, self.splits = training.train(
            train_df=self.train_df,
            cols_features=self._get_features(),
            clf_params=model_params,
            logger=self._logger,
            options=self._options,
        )

    def _normalize_and_cleanup(self, df):
        """
        Normalize the features and drop NaN-values if necessary.
        """
        features = self._get_features()
        df_features = df[features]

        std_scaler = StandardScaler()
        std_scaler.fit(df_features)
        df_features_scaled = pd.DataFrame(std_scaler.transform(df_features))

        df.loc[:, df_features_scaled.columns] = df_features_scaled

        return df

    def _true_random(self, min_val=0, max_val=2**32-1):
        state = random.getstate()
        random.seed(self._true_random_seed+self._true_random_ctr)
        self._true_random_ctr += 1
        val = random.randint(min_val, max_val)
        random.setstate(state)
        return val

    def _get_features(self):
        features_const = self._options['input']['columns']['features']
        feat_prefix = self._options['input']['columns']['feature_prefix']
        features_prefixes = [
            c for c in self.train_df.columns if str(c).startswith(feat_prefix)
        ]
        features = features_const + features_prefixes

        n_all_featrues = len(features)
        features = [
            f for f in features if not any(np.isnan(self.train_df[f].values))
        ]
        n_dropped_features = n_all_featrues - len(features)

        self._logger.info(f"Dropped {n_dropped_features} columns with NaN-values of {n_all_featrues}.")

        return features

    def rescore(self):
        """
        Run rescoring on input data.
        """
        self._logger.info('Start rescoring')
        cols_spectra = self._options['input']['columns']['spectrum_id']
        spectra_batch_size = self._options['rescoring']['spectra_batch_size']

        # Read spectra list
        spectra = readers.read_spectra_ids(
            self._input,
            cols_spectra,
            logger=self._logger,
            random_seed=self._true_random()
        )

        # Sort spectra
        spectra.sort()

        # Calculate number of batches
        n_batches = ceil(len(spectra)/spectra_batch_size)
        self._logger.info(f'Rescore in {n_batches} batches')

        # Iterate over spectra batches
        df_rescored = pd.DataFrame()
        for i_batch in range(n_batches):
            # Define batch borders
            spectra_range = spectra[
                i_batch*spectra_batch_size:(i_batch+1)*spectra_batch_size
            ]
            spectra_from = spectra_range[0]
            spectra_to = spectra_range[-1]
            self._logger.info(f'Start rescoring spectra batch {i_batch+1}/{n_batches} with `{spectra_from}` to `{spectra_to}`')

            # Read batch
            df_batch = readers.read_spectra_range(
                input=self._input,
                spectra_from=spectra_from,
                spectra_to=spectra_to,
                spectra_cols=cols_spectra,
                logger=self._logger,
                random_seed=self._true_random()
            )
            self._logger.info(f'Batch contains {len(df_batch)} samples')

            # Rescore batch
            df_batch = self.rescore_df(df_batch)

            # Store collected matches
            if type(self._output) is pd.DataFrame:
                df_rescored = pd.concat([
                    df_rescored,
                    df_batch
                ])
            else:
                writers.append_rescorings(
                    self._output,
                    df_batch,
                    options=self._options,
                    logger=self._logger,
                    random_seed=self._true_random()
                )

        # Keep rescored matches when no output is defined
        if type(self._output) is pd.DataFrame:
            self._output = pd.concat([
                self._output,
                df_rescored
            ])

    def rescore_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rescore a DataFrame of CSMs.

        :param df: CSMs to be rescored
        :type df: DataFrame

        :return: Rescored CSMs
        :rtype: DataFrame
        """
        cols_spectra = self._options['input']['columns']['spectrum_id']
        col_rescore = self._options['output']['columns']['rescore']
        col_top_ranking = self._options['input']['columns']['top_ranking']
        max_jobs = self._options['rescoring']['max_jobs']
        apply_logit = self._options['rescoring']['logit_result']
        if self._options['input']['columns']['csm_id'] is None:
            col_csm = list(self.train_df.columns)
        else:
            col_csm = self._options['input']['columns']['csm_id']


        # Normalize features
        df = self._normalize_and_cleanup(df)

        # Get feature columns
        feat_cols = self._get_features()

        # Rescore DF
        df_scores = rescoring.rescore(
            self.models,
            df=df[feat_cols],
            rescore_col=col_rescore,
            apply_logit=apply_logit,
            max_cpu=max_jobs
        )

        self._logger.info('Merge new scores into original data')

        df = df.merge(
            df_scores,
            left_index=True,
            right_index=True,
            validate='1:1',
        )

        # Rescore training data only with test fold classifier
        self._logger.info('Choose right score for training samples')
        df_slice = self.train_df.loc[:, col_csm].copy()
        df_slice[f'{col_rescore}_slice'] = -1
        for i, (_, idx_test) in enumerate(self.splits):
            df_slice.loc[idx_test, f'{col_rescore}_slice'] = i

        df = df.merge(
            df_slice,
            how='left',
            validate='1:1',
        )
        df.loc[
            df[f'{col_rescore}_slice'].isna(),
            f'{col_rescore}_slice'
        ] = -1

        df.loc[
            df[f'{col_rescore}_slice'] > -1,
            col_rescore
        ] = df.loc[
            df[f'{col_rescore}_slice'] > -1
        ].apply(
            _select_right_score,
            col_rescore=col_rescore,
            axis=1,
        )

        # Calculate top_ranking
        self._logger.info('Calculate top ranking scores')
        df_top_rank = df.groupby(cols_spectra).agg(max=(f'{col_rescore}', 'max')).rename(
            {'max': f'{col_rescore}_max'}, axis=1)
        df = df.merge(
            df_top_rank,
            left_on=list(cols_spectra),
            right_index=True
        )
        df.drop(col_top_ranking, axis=1, inplace=True, errors='ignore')
        df[col_top_ranking] = df[f'{col_rescore}'] == df[f'{col_rescore}_max']
        return df

    def get_rescored_output(self):
        if type(self._output) is pd.DataFrame:
            return self._output
        else:
            raise XiRescoreError('Not available for file or DB output.')


def _select_right_score(row, col_rescore):
    n_slice = int(row[f"{col_rescore}_slice"])
    return row[f'{col_rescore}_{n_slice}']


class XiRescoreError(Exception):
    """Custom exception for train data selection errors."""
    pass

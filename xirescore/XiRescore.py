"""Main module."""
import copy
import logging
import random
from collections.abc import Collection
from math import ceil

import numpy as np
import pandas as pd
from deepmerge import Merger
from sklearn.base import BaseEstimator, TransformerMixin

from xirescore import readers
from xirescore import rescoring
from xirescore import train_data_selecting
from xirescore import training
from xirescore import writers
from xirescore._default_options import default_options
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_extracting import get_features
from xirescore.feature_scaling import get_scaler
from xirescore.hyperparameter_optimizing import get_hyperparameters

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
    def __init__(self,
                 input_path,
                 output_path=None,
                 options=dict(),
                 logger=None,
                 loglevel=logging.DEBUG):
        """
        Initialize rescorer

        :param input_path: Path to input file/DB or an input DataFrame.
        :type input_path: str|DataFrame
        :param output_path: Path to the output file/DB or ``None`` if ``get_rescored_output()`` will be used.
        :type output_path: str, optional
        :param options: :ref:`options`
        :type options: dict, optional
        :param logger: Logger to be used. If ``None`` a new logger will be created.
        :type logger: Logger, optional
        :param loglevel: Log level to be used with new logger.
        :type loglevel: int, optional
        """
        # Apply override default options with user-supplied options
        self._options = copy.deepcopy(default_options)
        if 'model_params' in options.get('rescoring', dict()):
            # Discard default model_params if new ones are provided
            del self._options['rescoring']['model_params']
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
        if type(self._input) is pd.DataFrame:
            self._input = self._input.copy()
        if output_path is None:
            # Store output in new DataFrame if no path is given
            self._output = pd.DataFrame()
        else:
            self._output = output_path

        # Use supplied logger if present
        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._loglevel = loglevel
        self._true_random_ctr = 0

        self.train_df: pd.DataFrame
        """
        Data used for k-fold cross-validation.
        """
        self.splits: Collection[tuple[pd.Index, pd.Index]] = []
        """
        K-fold splits of model training. Kept to not rescore training samples with models they have been trained on.
        """
        self.models: list[BaseEstimator] = []
        """
        Trained models from the f-fold cross-validation.
        """
        self.scaler: TransformerMixin
        """
        Scaler for feature normalization.
        """
        self.train_features: list = []
        """
        Features extracted from training data.
        """

    def run(self) -> None:
        """
        Run training and rescoring of the input data and write to output
        """
        self._logger.info("Start full train and rescore run")
        self.train()
        self.rescore()

    def train(self,
              train_df: pd.DataFrame = None,
              splits: list[tuple[pd.Index, pd.Index]] = None):
        """
        Run training on input data or on the passed DataFrame if provided.

        :param train_df: Data to be used training instead of input data.
        :type train_df: DataFrame, optional
        :param splits: K-fold splits for manual ``train_df``.
        :type splits: Index, optional
        """
        self._logger.info('Start training')

        if train_df is None:
            self.train_df, self.scaler = train_data_selecting.select(
                self._input,
                self._options,
                self._logger
            )
        else:
            self.train_df = generate_columns(train_df, options=self._options, do_fdr=True, do_self_between=True)
            self.scaler = get_scaler(train_df, self._options, self._logger)

        if splits is not None:
            self.splits = splits

        self.train_features = get_features(self.train_df, self._options, self._logger)

        # Scale features
        self.train_df[self.train_features] = self.scaler.transform(
            self.train_df[self.train_features]
        )

        self._logger.info("Perform hyperparameter optimization")
        model_params = get_hyperparameters(
            train_df=self.train_df,
            cols_features=self.train_features,
            splits=splits,
            options=self._options,
            logger=self._logger,
            loglevel=self._loglevel,
        )

        self._logger.info("Train models")
        self.models, self.splits = training.train(
            train_df=self.train_df,
            cols_features=self.train_features,
            clf_params=model_params,
            splits=splits,
            logger=self._logger,
            options=self._options,
        )

    def get_rescoring_state(self) -> dict:
        """
        Get state of the current instance to use it later to recreate identical instance.

        :returns: Models and k-fold slices
        :rtype: dict
        """
        return {
            'splits': self.splits,
            'models': self.models,
        }

    def _true_random(self, min_val=0, max_val=2**32-1):
        state = random.getstate()
        random.seed(self._true_random_seed+self._true_random_ctr)
        self._true_random_ctr += 1
        val = random.randint(min_val, max_val)
        random.setstate(state)
        return val

    def rescore(self) -> None:
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
                sequence_p2_col=self._options['input']['columns']['base_sequence_p2'],
                only_pairs=True,
                logger=self._logger,
                random_seed=self._true_random()
            )
            self._logger.info(f'Batch contains {len(df_batch):,.0f} samples')
            self._logger.debug(f'Batch uses approx. {df_batch.memory_usage().sum()/1024/1024:,.2f}MB of RAM')

            # Rescore batch
            df_batch = self.rescore_df(df_batch)

            # Store collected matches
            self._logger.info('Write out batch')
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
            self._output = df_rescored

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

        # Scale features
        df[self.train_features] = self.scaler.transform(
            df[self.train_features]
        )

        # Rescore DF
        df_scores = rescoring.rescore(
            models=self.models,
            df=df[self.train_features],
            rescore_col=col_rescore,
            apply_logit=apply_logit,
            max_cpu=max_jobs
        )

        self._logger.info('Merge new scores into original data')

        # Rescore training data only with test fold classifier
        self._logger.info('Reconstruct training data slices')
        cols_merge = list(set(col_csm+cols_spectra))
        df_slice = self.train_df.loc[:, cols_merge].copy()
        df_slice[f'{col_rescore}_slice'] = -1
        for i, (_, idx_test) in enumerate(self.splits):
            df_slice.loc[idx_test, f'{col_rescore}_slice'] = i

        self._logger.info('Add merge columns to scores DataFrame')
        df_scores = pd.merge(
            df_scores,
            df,
            left_index=True,
            right_index=True,
            validate='1:1',
            suffixes=('_scores', '')
        )

        self._logger.info('Merge slice info into batch')
        df_scores['__index_backup__'] = df_scores.index
        df_scores = df_slice.merge(
            df_scores,
            on=cols_merge,
            how='right',
            validate='1:1',
        )
        df_scores.loc[
            df_scores[f'{col_rescore}_slice'].isna(),
            f'{col_rescore}_slice'
        ] = -1

        self._logger.info('Pick the correct score')
        df_scores.loc[
            df_scores[f'{col_rescore}_slice'] > -1,
            col_rescore
        ] = df_scores.loc[
            df_scores[f'{col_rescore}_slice'] > -1
        ].apply(
            _select_right_score,
            col_rescore=col_rescore,
            axis=1,
        )

        # Calculate top_ranking
        self._logger.info('Calculate top ranking scores')
        df_top_rank = df_scores.groupby(cols_spectra).agg(
            max=(f'{col_rescore}', 'max'),
            min=(f'{col_rescore}', 'min'),
        ).rename(
            {
                'max': f'{col_rescore}_max',
                'min': f'{col_rescore}_min',
            },
            axis=1
        ).reset_index()
        df_scores = df_scores.merge(
            df_top_rank,
            on=list(cols_spectra)
        )
        df_scores[f'{col_rescore}_rank'] = df_scores.groupby(cols_spectra)[col_rescore].rank(ascending=False)
        df_scores[f'{col_rescore}_{col_top_ranking}'] = df_scores[f'{col_rescore}'] == df_scores[f'{col_rescore}_max']
        df_scores.set_index('__index_backup__', inplace=True, drop=True)

        if getattr(self.scaler, 'inverse_transform', False):
            self._logger.info('Reverse scaling')
            df_scores[self.train_features] = self.scaler.inverse_transform(
                df_scores[self.train_features]
            )
        else:
            logging.warning('Scaler is missing ``inverse_transform()`` method')

        return df_scores

    def get_rescored_output(self) -> pd.DataFrame:
        """
        Get the rescoring results when no output was defined

        :returns: Rescoring results
        :rtype: DataFrame
        """
        if type(self._output) is pd.DataFrame:
            return self._output.reset_index(drop=True).copy()
        else:
            raise XiRescoreError('Not available for file or DB output.')


def _select_right_score(row, col_rescore):
    n_slice = int(row[f"{col_rescore}_slice"])
    return row[f'{col_rescore}_{n_slice}']


class XiRescoreError(Exception):
    """Custom exception for train data selection errors."""
    pass

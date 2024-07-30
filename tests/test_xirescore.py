#!/usr/bin/env python

"""Tests for `xirescore` package."""
import pandas as pd
import pytest
import logging
import tempfile
import subprocess
import os
import numpy as np
from xirescore.XiRescore import XiRescore


@pytest.mark.db
@pytest.mark.slow
def test_full_db_rescoring():
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DB rescoring test')
    options = {
        'input': {
            'columns': {
                'csm_id': [
                    'match_id',
                    'search_id'
                ]
            }
        },
        'rescoring': {
            'spectra_batch_size': 100_000
        }
    }
    rescorer = XiRescore(
        input_path='xi2resultsets://test:test@localhost:5432/xisearch2/fdbe9e59-2baa-44cb-b8cb-e8b7a590e136',
        output_path='xi2resultsets://test:test@localhost:5432/xisearch2',
        options=options,
        logger=logger,
    )
    rescorer.run()


@pytest.mark.parquet
def test_full_parquet_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full parquet rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000
            }
        }

        rescorer = XiRescore(
            input_path='./fixtures/test_data.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
            logger=logger,
        )
        rescorer.run()
        df_in = pd.read_parquet('./fixtures/test_data.parquet')
        df_out = pd.read_parquet(f'{tmpdirname}/result.parquet')
        assert len(df_in) == len(df_out)


@pytest.mark.parquet
@pytest.mark.svc
@pytest.mark.slow
def test_full_svc_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full parquet rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'model_class': 'svm',
                'model_name': 'SVC',
                'model_params': {
                    'kernel': ['rbf'],
                    'gamma': [1e-2, 1e-3, 1e-4, "auto"],
                    'C': [5, 8, 10, 15],
                    'probability': [True],
                    "class_weight": ["balanced", None, {0: 2, 1: 1}],
                    "tol": [100*np.finfo(np.double).eps]
                }
            }
        }

        rescorer = XiRescore(
            input_path='./fixtures/test_data.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
            logger=logger,
        )
        rescorer.run()
        df_in = pd.read_parquet('./fixtures/test_data.parquet')
        df_out = pd.read_parquet(f'{tmpdirname}/result.parquet')
        assert len(df_in) == len(df_out)


@pytest.mark.csv
def test_full_csv_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full CSV rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000  # Rescore in 4 batches
            }
        }

        rescorer = XiRescore(
            input_path='./fixtures/test_data.csv.gz',
            output_path=f'{tmpdirname}/result.csv.gz',
            options=options,
            logger=logger,
        )
        rescorer.run()
        df_in = pd.read_csv('./fixtures/test_data.csv.gz')
        df_out = pd.read_csv(f'{tmpdirname}/result.csv.gz')
        assert len(df_in) == len(df_out)


@pytest.mark.df
def test_full_df_rescoring():
    df = pd.read_parquet('./fixtures/test_data.parquet')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DF rescoring test')

    options = {
        'input': {
            'columns': {
                'features': [
                    'match_score',
                    'better_score',
                    'worse_score',
                    'useless_score_uni',
                    'useless_score_norm',
                    'conditional_score',
                ]
            }
        },
        'rescoring': {
            'spectra_batch_size': 25_000  # Rescore in 4 batches
        }
    }

    rescorer = XiRescore(
        input_path=df,
        options=options,
        logger=logger,
    )
    rescorer.run()
    df_out = rescorer.get_rescored_output()
    assert len(df) == len(df_out)


@pytest.mark.cli
def test_full_cli_parquet_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full CSV rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000  # Rescore in 4 batches
            }
        }

        result = subprocess.run(
            [
                'xirescore',
                '-i', './fixtures/test_data.parquet',
                '-o', f'{tmpdirname}/result.parquet',
                '-C', str(options),
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
        assert os.path.exists(f'{tmpdirname}/result.parquet'), f"Output file was not created."

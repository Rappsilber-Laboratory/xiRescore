#!/usr/bin/env python

"""Tests for `xirescore` package."""

import pytest
import logging
import tempfile

from xirescore.XiRescore import XiRescore


@pytest.mark.db
def test_full_db_rescoring():
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DB rescoring test')
    rescorer = XiRescore(
        input_path='xi2resultsets://test:test@localhost:5432/xisearch2/fdbe9e59-2baa-44cb-b8cb-e8b7a590e136',
        output_path='xi2resultsets://test:test@localhost:5432/xisearch2',
        logger=logger
    )
    rescorer.run()


def test_full_parquet_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logger = logging.getLogger(__name__)
        logging.basicConfig()
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
                'spectra_batch_size': 25_000  # Rescore in 4 batches
            }
        }

        rescorer = XiRescore(
            input_path='./fixtures/test_data.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
            logger=logger,
        )
        rescorer.run()


def test_cli():
    raise Exception('Not implemented yet.')

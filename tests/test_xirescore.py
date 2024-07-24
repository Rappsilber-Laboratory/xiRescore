#!/usr/bin/env python

"""Tests for `xirescore` package."""

import pytest
import logging

from xirescore.XiRescore import XiRescore


def test_full_db_rescoring():
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DB rescoring test')
    rescorer = XiRescore(
        input_path='xi2resultsets://test:test@localhost:5432/xisearch2/fdbe9e59-2baa-44cb-b8cb-e8b7a590e136',
        output_path='xi2resultsets://test:test@localhost:5432/xisearch2/',
        logger=logger
    )
    rescorer.run()

def test_cli():
    raise Exception('Not implemented yet.')

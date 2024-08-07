#!/usr/bin/env python

"""Tests for `xirescore` package."""
import pandas as pd
import logging
import numpy as np
import random
from xirescore.XiRescore import XiRescore
import tempfile


def test_merge_samples():
    """
    Check if samples are correctly merged and if trivial classification works.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    random.seed(0)
    np.random.seed(0)

    n_samples = 500_000

    df = pd.DataFrame(index=range(n_samples))
    df['is_decoy_p1'] = np.round(np.random.uniform(size=n_samples)).astype(bool)
    df['is_decoy_p2'] = np.round(np.random.uniform(size=n_samples)).astype(bool)
    df['self'] = np.round(np.random.uniform(size=n_samples)).astype(bool)
    df['base_sequence_p1'] = [
        ''.join(random.choices(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], k=20))
        for _ in range(n_samples)
    ]
    df['base_sequence_p2'] = [
        ''.join(random.choices(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'], k=20))
        for _ in range(n_samples)
    ]
    df['fdr_group'] = 'between'
    df.loc[df['self'], 'fdr_group'] = 'self'
    df['csm_id'] = df.index/len(df)
    df['spectrum_id'] = df.index
    df['top_ranking'] = True
    df['isTT'] = (~df['is_decoy_p1']) & (~df['is_decoy_p2'])
    df['feature_idx'] = df['csm_id']
    df['feature_isTT'] = df['isTT']

    n_tt = len(df[df['isTT']])
    n_dx = len(df[~df['isTT']])
    df['match_score'] = 0
    df.loc[
        df['isTT'],
        'feature_match_score'
    ] = np.random.normal(loc=22, size=n_tt)
    df.loc[
        ~df['isTT'],
        'feature_match_score'
    ] = np.random.normal(loc=20, size=n_dx)

    logger.info('Start full DF rescoring test')

    options = {
        'input': {
            'columns': {
                'csm_id': ['csm_id']
            }
        }
    }
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        df.to_csv(f'{tmpdirname}/input.csv.gz')
        rescorer = XiRescore(
            input_path=f'{tmpdirname}/input.csv.gz',
            output_path=f'{tmpdirname}/output.csv.gz',
            options=options,
            logger=logger,
        )
        rescorer.run()
        df_out = pd.read_csv(f'{tmpdirname}/output.csv.gz')
        assert len(df) == len(df_out)
        assert np.count_nonzero(
            np.isclose(df_out['isTT'], df_out['feature_isTT'])
        ) == len(df_out)
        assert np.count_nonzero(
            np.isclose(df_out['csm_id'], df_out['feature_idx'])
        ) == len(df_out)
        classifications = df_out['rescore'] > 0
        assert np.count_nonzero(
            df_out['isTT'] == classifications
        ) == len(df_out)

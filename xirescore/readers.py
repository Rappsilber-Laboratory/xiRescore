"""
Readers for data inputs
"""
import random
from math import ceil
from typing import Union
from collections.abc import Sequence

import numpy as np
import pandas as pd
from fastparquet import ParquetFile as FPParquetFile
from pyarrow.parquet import ParquetDataset as PAParquetDataset
import pyarrow.compute as pc
import pyarrow

from xirescore.DBConnector import DBConnector


_dbs = dict()


def _get_db(hostname,
           port,
           username,
           password,
           database,
           logger=None,
           random_seed=random.randint(0, 2**32-1)):
    if (hostname, port, username, password, database) not in _dbs:
        _dbs[
            (hostname, port, username, password, database)
        ] = DBConnector(
            username=username,
            password=password,
            hostname=hostname,
            port=port,
            database=database,
            logger=logger,
            random_seed=random_seed,
        )
    return _dbs[
        (hostname, port, username, password, database)
    ]


def read_spectra_ids(path, spectra_cols=None, logger=None, random_seed=None) -> pd.DataFrame:
    if type(path) is pd.DataFrame:
        return path.loc[:, spectra_cols]\
            .drop_duplicates()\
            .to_records(index=False)

    file_type = get_source_type(path)

    if file_type != 'db' and spectra_cols is None:
        raise ValueError('Filetype {file_type} requires parameter `spectra_cols`!')

    if file_type == 'csv':
        return pd.read_csv(path, usecols=spectra_cols) \
            .loc[:, spectra_cols] \
            .drop_duplicates() \
            .to_records(index=False)
    if file_type == 'tsv':
        return pd.read_csv(path, sep='\t', usecols=spectra_cols) \
            .loc[:, spectra_cols] \
            .drop_duplicates() \
            .to_records(index=False)
    if file_type == 'parquet':
        return pd.read_parquet(path, columns=spectra_cols) \
            .loc[:, spectra_cols] \
            .drop_duplicates() \
            .to_records(index=False)
    if file_type == 'db':
        db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(path)
        db = _get_db(
            username=db_user,
            password=db_pass,
            hostname=db_host,
            port=db_port,
            database=db_db,
            logger=logger,
            random_seed=random_seed
        )
        return db.read_spectrum_ids(resultset_ids=rs_ids) \
            .loc[:, ['spectrum_id']] \
            .drop_duplicates() \
            .to_records(index=False)


def read_spectra_range(input: Union[str, pd.DataFrame],
                       spectra_from: Sequence[Sequence],
                       spectra_to: Sequence[Sequence],
                       spectra_cols: Sequence = None,
                       sequence_p2_col='sequence_p2',
                       only_pairs=True,
                       logger=None,
                       random_seed=None):
    # Handle input DF
    if type(input) is pd.DataFrame:
        filters = (
            (input[spectra_cols].apply(lambda r: tuple(r), axis=1) >= tuple(spectra_from)) &
            (input[spectra_cols].apply(lambda r: tuple(r), axis=1) <= tuple(spectra_to))
        )
        if only_pairs:
            filters &= ~input[sequence_p2_col].isna()
            filters &= input[sequence_p2_col] != ''
        return input[filters].copy()
    # Handle input path
    file_type = get_source_type(input)
    if file_type == 'csv':
        return read_spectra_range_csv(
            input,
            spectra_from,
            spectra_to,
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
        )
    if file_type == 'tsv':
        return read_spectra_range_csv(
            input,
            spectra_from,
            spectra_to,
            sep='\t',
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
        )
    if file_type == 'db':
        return read_spectra_range_db(
            input,
            spectra_from,
            spectra_to,
            logger=logger,
            random_seed=random_seed,
            only_pairs=only_pairs,
        )
    if file_type == 'parquet':
        return read_spectra_range_parquet(
            input,
            spectra_from,
            spectra_to,
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
        )


def read_spectra_db(path, spectra: Sequence[Sequence], random_seed=None):
    db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(path)
    db = _get_db(
        username=db_user,
        password=db_pass,
        hostname=db_host,
        port=db_port,
        database=db_db,
        random_seed=random_seed
    )
    return db.read_resultsets(
        resultset_ids=rs_ids,
        spectrum_ids=[
            s[0] for s in spectra
        ],
        only_pairs=True,
    )


def read_spectra_range_db(path, spectra_from, spectra_to, only_pairs, logger, random_seed=None):
    db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(path)
    db = _get_db(
        username=db_user,
        password=db_pass,
        hostname=db_host,
        port=db_port,
        database=db_db,
        random_seed=random_seed,
        logger=logger,
    )
    return db.read_spectra_range(
        resultset_ids=rs_ids,
        spectra_from=spectra_from[0],
        spectra_to=spectra_to[0],
        only_pairs=only_pairs,
    )


def read_spectra_parquet(path, spectra: Sequence[Sequence], spectra_cols: Sequence):
    # Filters for spectrum columns
    filters = [
        [
            (spectra_col, 'in', spectrum[col_i])
            for col_i, spectra_col in enumerate(spectra_cols)
        ]
        for spectrum in spectra
    ]

    df = pd.read_parquet(path, filters=filters)
    return df.reset_index(drop=True)


def read_spectra_range_parquet(path,
                               spectra_from,
                               spectra_to,
                               spectra_cols: Sequence,
                               sequence_p2_col='sequence_p2',
                               only_pairs=True):
    # Filters for spectrum columns
    parquet_file = FPParquetFile(path)
    res_df = pd.DataFrame()
    for df in parquet_file.iter_row_groups():
        # Type-hint
        df: pd.DataFrame
        # Generate filters
        filters = (
            (df[spectra_cols].apply(lambda r: tuple(r), axis=1) >= tuple(spectra_from)) &
            (df[spectra_cols].apply(lambda r: tuple(r), axis=1) <= tuple(spectra_to))
        )
        # Filter out linear matches
        if only_pairs:
            filters &= ~df[sequence_p2_col].isna()
            filters &= df[sequence_p2_col] != ''
        # Append row group
        res_df = pd.concat(
            [
                res_df,
                df[filters]
            ]
        )
    return res_df.reset_index(drop=True).copy()


def read_spectra_csv(path, spectra: Sequence[Sequence], spectra_cols: Sequence, sep=',', chunksize=500_000):
    # Initialize result DataFrame
    res_df = pd.DataFrame()
    for df in pd.read_csv(path, sep=sep, chunksize=chunksize):
        filters = False
        # Generate filters for the requested spectra
        for spectrum in spectra:
            sub_filter = True
            for col_i, spectra_col in enumerate(spectra_cols):
                sub_filter &= df[spectra_col] == spectrum[col_i]
            filters |= sub_filter
        # Append filtered chunk
        res_df = pd.concat(
            [
                res_df,
                df[filters]
            ]
        )
    return res_df.reset_index(drop=True).copy()


def read_spectra_range_csv(path,
                           spectra_from,
                           spectra_to,
                           spectra_cols: Sequence,
                           sequence_p2_col='sequence_p2',
                           only_pairs=True,
                           sep=',',
                           chunksize=500_000):
    # Initialize result DataFrame
    res_df = pd.DataFrame()
    for df in pd.read_csv(path, sep=sep, chunksize=chunksize):
        # Generate filters for the requested spectra range
        filters = (
            (df[spectra_cols].apply(lambda r: tuple(r), axis=1) >= tuple(spectra_from)) &
            (df[spectra_cols].apply(lambda r: tuple(r), axis=1) <= tuple(spectra_to))
        )
        # Filter out linear matches
        if only_pairs:
            filters &= ~df[sequence_p2_col].isna()
            filters &= df[sequence_p2_col] != ''
        # Append filtered chunk
        res_df = pd.concat(
            [
                res_df,
                df[filters]
            ]
        )
    return res_df.reset_index(drop=True).copy()


def get_source_type(path: str):
    if path.lower().endswith('.parquet') or path.lower().endswith('.parquet/'):
        return 'parquet'
    if path.startswith('xi2resultsets://'):
        return 'db'
    if path.lower().endswith('.tsv') or path.lower().endswith('.tab'):
        return 'tsv'
    if path.lower().endswith('.csv'):
        return 'csv'
    if len(path.split('.')) > 2:
        ext2 = path.split('.')[-2].lower()
        if ext2 == 'csv':
            return 'csv'
        if ext2 == 'tab' or ext2 == 'tsv':
            return 'tsv'
    raise ValueError(f'Unknown file type of {path}')


def parse_db_path(path):
    db_no_prot = path.replace('xi2resultsets://', '')
    db_conn, db_path = db_no_prot.split('/', maxsplit=1)
    db_db, rs_ids = db_path.split('/', maxsplit=1)
    db_auth, db_tcp = db_conn.split('@', maxsplit=1)
    db_user, db_pass = db_auth.split(':', maxsplit=1)
    db_host, db_port = db_tcp.split(':', maxsplit=1)
    rs_ids = rs_ids.split(';')
    return db_user, db_pass, db_host, db_port, db_db, rs_ids


def read_sample(input_data,
                sample=1_000_000,
                top_ranking_col='top_ranking',
                sequence_p2_col='sequence_p2',
                logger=None,
                only_top_ranking=False,
                only_pairs=True,
                random_seed=None):
    if type(input_data) is pd.DataFrame:
        filter = np.repeat(True, repeats=len(input_data))
        if only_top_ranking:
            filter &= input_data[top_ranking_col]
        if only_pairs:
            filter &= ~input_data[sequence_p2_col].isna()
            filter &= input_data[sequence_p2_col] != ''
        sample_min = min(
            len(input_data[filter]),
            sample
        )
        return input_data[filter].sample(sample_min)
    file_type = get_source_type(input_data)
    if file_type == 'csv':
        return read_sample_csv(
            input_data,
            sample=sample,
            top_ranking_col=top_ranking_col,
            sequence_p2_col=sequence_p2_col,
            only_top_ranking=only_top_ranking,
            only_pairs=only_pairs,
        )
    if file_type == 'tsv':
        return read_sample_csv(
            input_data,
            sep='\t',
            sample=sample,
            top_ranking_col=top_ranking_col,
            only_top_ranking=only_top_ranking,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
        )
    if file_type == 'db':
        db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(input_data)
        db = _get_db(
            username=db_user,
            password=db_pass,
            hostname=db_host,
            port=db_port,
            database=db_db,
            random_seed=random_seed,
            logger=logger,
        )
        return db.read_resultsets(
            resultset_ids=rs_ids,
            only_pairs=only_pairs,
            only_top_ranking=only_top_ranking,
            sample=sample,
        )
    if file_type == 'parquet':
        return read_sample_parquet(
            input_data,
            sample=sample,
            sequence_p2_col=sequence_p2_col,
            only_top_ranking=only_top_ranking,
            top_ranking_col=top_ranking_col,
            only_pairs=only_pairs,
        )


def read_sample_parquet(path: str,
                        sample: int,
                        top_ranking_col='top_ranking',
                        sequence_p2_col='sequence_p2',
                        batch_size=500_000,
                        only_top_ranking=False,
                        only_pairs=True,
                        random_state=random.randint(0, 2**32-1)):
    parquet_file = PAParquetDataset(path)
    n_parts = 0
    for f in parquet_file.fragments:
        f: pyarrow.dataset.ParquetFileFragment
        n_parts += f.num_row_groups
    res_df = pd.DataFrame()
    filter = pc.scalar(True)
    if only_top_ranking:
        filter &= pc.field(top_ranking_col)
    if only_pairs:
        filter &= ~pc.is_null(pc.field(sequence_p2_col))
        filter &= pc.not_equal(pc.field(sequence_p2_col), '')
    for frag in parquet_file.fragments:
        for batch in frag.to_batches(filter=filter, batch_size=batch_size):
            df = batch.to_pandas()
            df: pd.DataFrame
            n_group_samples = min(
                int(sample / n_parts),
                len(df)
            )
            res_df = pd.concat(
                [
                    res_df,
                    df.sample(n_group_samples, random_state=random_state)
                ]
            )
    return res_df.reset_index(drop=True)


def read_sample_csv(path,
                    sample,
                    sep=',',
                    chunksize=5_000_000,
                    top_ranking_col='top_ranking',
                    sequence_p2_col='sequence_p2',
                    only_pairs=True,
                    only_top_ranking=False,
                    random_state=random.randint(0, 2**32-1)):
    n_rows = sum(1 for _ in open(path, 'rb'))
    n_chunks = ceil(n_rows / sample)
    res_df = pd.DataFrame()
    for i_chunk, df in enumerate(pd.read_csv(path, sep=sep, chunksize=chunksize)):
        filter = np.repeat(True, repeats=len(df))
        if only_top_ranking:
            filter &= df[top_ranking_col]
        if only_pairs:
            filter &= ~df[sequence_p2_col].isna()
            filter &= df[sequence_p2_col] != ''
        df = df[filter]
        res_df = pd.concat([
            res_df,
            df
        ])
        # How much of the chunks have been processed? (+1 to be safe)
        prop_chunks = (i_chunk + 1) / n_chunks
        subsample_size = min(
            len(res_df),
            int(sample * prop_chunks),
        )
        res_df = res_df.sample(subsample_size, random_state=random_state)
    final_sample = min(
        sample,
        len(res_df)
    )
    return res_df.sample(final_sample, random_state=random_state).reset_index(drop=True)

import pandas as pd
from xirescore.DBConnector import DBConnector
from collections.abc import Sequence
from fastparquet import ParquetFile as FPParquetFile
from pyarrow.parquet import ParquetFile as PAParquetFile
from math import ceil
import multiprocess as mp
from time import sleep


def read_spectra_ids(path, spectra_cols=None) -> pd.DataFrame:
    if type(path) is pd.DataFrame:
        return path.loc[:, spectra_cols].drop_duplicates()

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
        db = DBConnector(
            username=db_user,
            password=db_pass,
            hostname=db_host,
            port=db_port,
            database=db_db
        )
        return db.read_spectrum_ids(resultset_ids=rs_ids) \
            .loc[:, ['spectrum_id']] \
            .drop_duplicates() \
            .to_records(index=False)


def read_spectra(path: str, spectra: Sequence[Sequence], spectra_cols: Sequence):
    file_type = get_source_type(path)
    if file_type == 'csv':
        return read_spectra_csv(path, spectra, spectra_cols=spectra_cols)
    if file_type == 'tsv':
        return read_spectra_csv(path, spectra, sep='\t', spectra_cols=spectra_cols)
    if file_type == 'db':
        return read_spectra_db(path, spectra)
    if file_type == 'parquet':
        return read_spectra_parquet(path, spectra, spectra_cols=spectra_cols)


def read_spectra_range(input: str | pd.DataFrame,
                       spectra_from: Sequence[Sequence],
                       spectra_to: Sequence[Sequence],
                       spectra_cols: Sequence = None,
                       logger=None):
    # Handle input DF
    if type(input) is pd.DataFrame:
        filters = input[
            (input[spectra_cols].apply(lambda r: tuple(r)) >= spectra_from) &
            (input[spectra_cols].apply(lambda r: tuple(r)) <= spectra_to)
        ]
        return input[filters]
    # Handle input path
    file_type = get_source_type(input)
    if file_type == 'csv':
        return read_spectra_range_csv(input, spectra_from, spectra_to, spectra_cols=spectra_cols)
    if file_type == 'tsv':
        return read_spectra_range_csv(input, spectra_from, spectra_to, sep='\t', spectra_cols=spectra_cols)
    if file_type == 'db':
        return read_spectra_range_db(input, spectra_from, spectra_to, logger=logger)
    if file_type == 'parquet':
        return read_spectra_range_parquet(input, spectra_from, spectra_to, spectra_cols=spectra_cols)


def get_spectra_range_queue(input: str | pd.DataFrame,
                            ranges,
                            spectra_cols: Sequence = None,
                            logger=None):
    res_queue = mp.Queue()
    p = mp.Process(
        target=spectra_range_prefetcher,
        args=(input, ranges, res_queue, spectra_cols, logger)
    )
    p.start()
    return res_queue


def spectra_range_prefetcher(input: str | pd.DataFrame,
                             ranges,
                             res_queue: mp.Queue,
                             spectra_cols: Sequence = None,
                             logger=None):
    for spectra_from, spectra_to in ranges:
        while not res_queue.empty():
            sleep(1)
        res = read_spectra_range(
            input=input,
            spectra_from=spectra_from,
            spectra_to=spectra_to,
            spectra_cols=spectra_cols,
            logger=logger
        )
        res_queue.put(res)


def read_spectra_db(path, spectra: Sequence[Sequence]):
    db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(path)
    db = DBConnector(db_user, db_pass, db_host, db_port, db_db)
    return db.read_resultsets(
        resultset_ids=rs_ids,
        spectrum_ids=[
            s[0] for s in spectra
        ],
        only_pairs=True,
    )


def read_spectra_range_db(path, spectra_from, spectra_to, logger):
    db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(path)
    db = DBConnector(
        username=db_user,
        password=db_pass,
        hostname=db_host,
        port=db_port,
        database=db_db,
        logger=logger,
    )
    return db.read_spectra_range(
        resultset_ids=rs_ids,
        spectra_from=spectra_from[0],
        spectra_to=spectra_to[0],
        only_pairs=True,
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
    return df


def read_spectra_range_parquet(path,
                               spectra_from,
                               spectra_to,
                               spectra_cols: Sequence):
    # Filters for spectrum columns
    parquet_file = FPParquetFile(path)
    res_df = pd.DataFrame()
    for df in parquet_file.iter_row_groups():
        # Type-hint
        df: pd.DataFrame
        # Generate filters
        filters = (
            (df[spectra_cols].apply(lambda r: tuple(r)) >= tuple(spectra_from)) &
            (df[spectra_cols].apply(lambda r: tuple(r)) <= tuple(spectra_to))
        ).iloc[:, 0]
        # Append row group
        res_df = pd.concat(
            [
                res_df,
                df[filters]
            ]
        )
    return res_df.copy()


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
    return res_df.copy()


def read_spectra_range_csv(path,
                           spectra_from,
                           spectra_to,
                           spectra_cols: Sequence,
                           sep=',',
                           chunksize=500_000):
    # Initialize result DataFrame
    res_df = pd.DataFrame()
    for df in pd.read_csv(path, sep=sep, chunksize=chunksize):
        # Generate filters for the requested spectra range
        filters = (
            (df[spectra_cols].apply(lambda r: tuple(r)) >= tuple(spectra_from)) &
            (df[spectra_cols].apply(lambda r: tuple(r)) <= tuple(spectra_to))
        ).iloc[:, 0]
        # Append filtered chunk
        res_df = pd.concat(
            [
                res_df,
                df[filters]
            ]
        )
    return res_df.copy()


def get_source_type(path: str):
    if path.lower().endswith('.parquet'):
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


def read_top_sample(input_data,
                    sample=1_000_000,
                    top_ranking_col='top_ranking',
                    logger=None):
    if type(input_data) is pd.DataFrame:
        sample_min = min(
            len(input_data),
            sample
        )
        return input_data[
            input_data[top_ranking_col]
        ].sample(sample_min)
    file_type = get_source_type(input_data)
    if file_type == 'csv':
        return read_top_sample_csv(input_data, sample=sample, top_ranking_col=top_ranking_col)
    if file_type == 'tsv':
        return read_top_sample_csv(input_data, sep='\t', sample=sample, top_ranking_col=top_ranking_col)
    if file_type == 'db':
        db_user, db_pass, db_host, db_port, db_db, rs_ids = parse_db_path(input_data)
        db = DBConnector(
            username=db_user,
            password=db_pass,
            hostname=db_host,
            port=db_port,
            database=db_db,
            logger=logger,
        )
        return db.read_resultsets(
            resultset_ids=rs_ids,
            only_pairs=True,
            only_top_ranking=True,
            sample=sample,
        )
    if file_type == 'parquet':
        return read_top_sample_parquet(input_data, sample=sample, top_ranking_col=top_ranking_col)


def read_top_sample_parquet(path, sample, top_ranking_col='top_ranking', random_state=0):
    n_row_groups = PAParquetFile(path).num_row_groups
    parquet_file = FPParquetFile(path)
    res_df = pd.DataFrame()
    filters = [[(top_ranking_col, '==', True)]]
    for df in parquet_file.iter_row_groups(filters=filters):
        df: pd.DataFrame
        n_group_samples = min(
            sample / n_row_groups,
            len(df)
        )
        res_df = pd.concat(
            [
                res_df,
                df.sample(n_group_samples, random_state=random_state)
            ]
        )
    return res_df


def read_top_sample_csv(path, sample, sep=',', chunksize=5_000_000, top_ranking_col='top_ranking', random_state=0):
    n_rows = sum(1 for _ in open(path, 'rb'))
    n_chunks = ceil(n_rows / sample)
    res_df = pd.DataFrame()
    for i_chunk, df in enumerate(pd.read_csv(path, sep=sep, chunksize=chunksize)):
        df = df[
            df[top_ranking_col]
        ]
        res_df = pd.concat([
            res_df,
            df
        ])
        # How much of the chunks have been processed? (+1 to be safe)
        prop_chunks = (i_chunk + 1) / n_chunks
        subsample_size = min(
            len(res_df),
            sample * prop_chunks,
        )
        res_df = res_df.sample(subsample_size, random_state=random_state)
    final_sample = min(
        sample,
        len(res_df)
    )
    return res_df.sample(final_sample, random_state=random_state)

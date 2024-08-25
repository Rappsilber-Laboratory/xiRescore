"""
Writers for data outputs
"""
from pathlib import Path
import random
import os

import fastparquet
import pandas as pd

from xirescore.readers import get_source_type
from xirescore.df_serializing import serialize_columns
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


def append_rescorings(output,
                      df: pd.DataFrame,
                      options=dict(),
                      logger=None,
                      random_seed=random.randint(0,2**32-1)):
    output_type = get_source_type(output)
    if output_type == 'csv':
        append_csv(output, df)
    if output_type == 'tsv':
        append_csv(output, df, sep='\t')
    if output_type == 'parquet':
        append_parquet(output, df)
    if output_type == 'db':
        append_db(
            output=output,
            df=df,
            options=options,
            logger=logger,
            random_seed=random_seed,
        )


def append_parquet(output, df: pd.DataFrame, compression='GZIP'):
    df = serialize_columns(df)
    fastparquet.write(
        output,
        data=df,
        compression=compression,
        write_index=False,
        append=Path(output).is_file(),
    )


def append_csv(output, df: pd.DataFrame, sep=','):
    df.to_csv(
        output,
        mode='a',
        header=not os.path.isfile(output),
        sep=sep
    )


def parse_db_path(path):
    db_no_prot = path.replace('xi2resultsets://', '')
    db_conn, db_db = db_no_prot.split('/', maxsplit=1)
    db_auth, db_tcp = db_conn.split('@', maxsplit=1)
    db_user, db_pass = db_auth.split(':', maxsplit=1)
    db_host, db_port = db_tcp.split(':', maxsplit=1)
    return db_user, db_pass, db_host, db_port, db_db


def append_db(output,
              df: pd.DataFrame,
              options,
              logger=None,
              random_seed=random.randint(0, 2**32-1)):
    col_rescore = options['output']['columns']['rescore']

    db_user, db_pass, db_host, db_port, db_db = parse_db_path(output)
    db = _get_db(
        username=db_user,
        password=db_pass,
        hostname=db_host,
        port=db_port,
        database=db_db,
        logger=logger,
        random_seed=random_seed,
    )
    cols_scores = [
        c for c in df.columns
        if str(c).startswith(f'{col_rescore}_') or c == col_rescore
    ]
    last_resultset_id_written = db.get_last_resultset_id_written()
    if last_resultset_id_written is None:
        # Create new resultset
        score_names = [c for c in df.columns if str(c).startswith(col_rescore)]
        search_ids = df['search_id'].drop_duplicates().values
        resultset_name_join = ';'.join(
            df['resultset_name'].drop_duplicates().values
        )
        resultset_name = f'xiRescore({resultset_name_join})'
        resultset_id = db.create_resultset(
            resultset_name,
            score_names=score_names,
            main_score=col_rescore,
            config=str(options),
            search_ids=search_ids,
        )
        logger.info(f'Create resultset `{resultset_id}`')
    else:
        resultset_id = last_resultset_id_written
    rescore_base_name = options['output']['columns']['rescore']
    db.write_resultmatches(
        df,
        feature_cols=cols_scores,
        resultset_id=resultset_id,
        top_ranking_col=f'{rescore_base_name}_top_ranking'
    )

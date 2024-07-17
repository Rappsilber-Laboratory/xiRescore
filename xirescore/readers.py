import pandas as pd
import DBConnector

def read_spectra(path: str, spectra: list, spectra_cols: list):
    file_type = get_source_type(path, spectra_cols=spectra_cols)
    if file_type == 'csv':
        return read_spectra_csv(path, spectra, spectra_cols=spectra_cols)
    if file_type == 'tsv':
        return read_spectra_csv(path, spectra, sep='\t', spectra_cols=spectra_cols)
    if file_type == 'db':
        return read_spectra_db(path, spectra, spectra_cols=spectra_cols)
    if file_type == 'parquet':
        return read_spectra_parquet(path, spectra)


def read_spectra_db(path, spectra: list):
    pass


def read_spectra_parquet(path, spectra: list, spectra_cols: list):
    if len(spectra_cols) == 1:
        # Filters for a single spectrum column
        spectra_col = spectra_cols[0]
        filters = [
            [(spectra_col, 'in', spectrum)]
            for spectrum in spectra
        ]
    else:
        # Filters for multiple spectrum columns
        filters = [
            [
                (spectra_col, 'in', spectrum[col_i])
                for col_i, spectra_col in enumerate(spectra_cols)
            ]
            for spectrum in spectra
        ]

    df = pd.read_parquet(path, filters=filters)
    return df


def read_spectra_csv(path, spectra: list, spectra_cols: list, sep=',', chunksize=500_000):
    # Initialize result DataFrame
    res_df = pd.DataFrame()
    for df in pd.read_csv(path, sep=sep, chunksize=chunksize):
        filters = False
        # Generate filters for the requested spectra
        for spectrum in spectra:
            if len(spectra_cols) == 1:
                spectra_col = spectra_cols[0]
                filters |= df[spectra_col] == spectrum
            else:
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
    return db_user, db_pass, db_host, db_port, db_db, rs_ids

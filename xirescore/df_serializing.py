from collections.abc import Iterable
import uuid

import numpy as np


def serialize_columns(df):
    nan_filter = None
    for c in df.columns:
        if nan_filter is None:
            nan_filter = ~df[c].isna()
        else:
            nan_filter &= ~df[c].isna()

    # Try to find row without NaNs
    type_row = df[nan_filter]

    if len(type_row) > 0:
        # If some found take the first one
        type_row = type_row.iloc[:1]
    else:
        # If none found take the first row od the DF
        type_row = df.iloc[:1]

    for c in df.columns:
        col_row = type_row
        # Check if type_row has a valid value for column
        if col_row.iloc[0][c] is None:
            # If not search for a row with a non-NaN value
            col_row = df[~df[c].isna()]
            if len(col_row) == 0:
                # Only NaN values for column
                continue
        # Take the non-NaN value found
        col_row = col_row.iloc[0]
        if type(col_row[c]) is uuid.UUID:
            df[df[c].isna()] = ''
            df[c] = df[c].astype(str, errors='ignore')
        elif type(col_row[c]) is str:
            pass
        elif isinstance(col_row[c], Iterable):
            df[df[c].isna()] = ''
            df[c] = df[c].apply(
                lambda x: None if x is None else ';'.join(np.array(x).astype(str))
            )
        elif np.issubdtype(type(col_row[c]), np.generic):
            pass
        else:
            df[df[c].isna()] = ''
            df[c] = df[c].astype(str, errors='ignore')
    df.columns = df.columns.map(str)
    return df

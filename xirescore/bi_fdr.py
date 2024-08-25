import os
from functools import partial
import multiprocess as mp
from math import ceil
from time import sleep

import pandas as pd
import numpy as np


def self_or_between(df,
                    col_prot1='protein_p1',
                    col_prot2='protein_p2',
                    str_self='self',
                    str_between='between',
                    decoy_adj='REV_'):
    df.loc[:, f'{col_prot1}_arr'] = df.loc[:, col_prot1]\
        .astype(str).str.replace(decoy_adj, '')\
        .str.split(';').map(np.unique).map(list)
    df.loc[:, f'{col_prot2}_arr'] = df.loc[:, col_prot2]\
        .astype(str).str.replace(decoy_adj, '')\
        .str.split(';').map(np.unique).map(list)
    df.loc[:, 'proteins_arr'] = (df.loc[:, f'{col_prot1}_arr'] + df.loc[:, f'{col_prot2}_arr'])
    df.loc[:, 'proteins_arr_unique'] = df.loc[:, 'proteins_arr'].map(np.unique)
    df.loc[:, 'is_between'] = ~(
            df.loc[:, 'proteins_arr'].map(len) - df.loc[:, 'proteins_arr_unique'].map(len)
    ).astype(bool)
    df.drop(
        [
            f'{col_prot1}_arr',
            f'{col_prot2}_arr',
            'proteins_arr',
            'proteins_arr_unique'
        ],
        axis=1,
        inplace=True
    )
    return df.loc[:, 'is_between'].map({True: str_between, False: str_self})


def self_or_between_mp(df, col_prot1='protein_p1', col_prot2='protein_p2', decoy_adj='Rev_'):
    pool_size = min([10,os.cpu_count()])
    slice_size = ceil(len(df)/pool_size)
    cols = [col_prot1, col_prot2]
    df = df.loc[:, cols]
    df_slices = [
        df.iloc[i*slice_size:(i+1)*slice_size][cols]
        for i in range(pool_size)
    ]
    print('slicing done')
    print(f"Pool size: {pool_size}")
    with mp.Pool(processes=pool_size) as pool:
        job_self_between = partial(
            self_or_between,
            decoy_adj=decoy_adj,
            col_prot1=col_prot1,
            col_prot2=col_prot2,
        )
        map_res = pool.map(job_self_between, df_slices)
    return pd.concat(map_res).copy()


def calculate_fdr(df,
                  decoy_p1='decoy_p1',
                  decoy_p2='decoy_p2',
                  score_col='match_score',
                  decoy_class=None,
                  max_slices=None):
    orig_sorting = df.index
    df_by_score_desc = df.sort_values(score_col, ascending=False)

    if decoy_class is None:
        is_dd = df_by_score_desc.loc[:, decoy_p1] & df_by_score_desc.loc[:, decoy_p2]
        is_tt = (~df_by_score_desc.loc[:, decoy_p1]) & (~df_by_score_desc.loc[:, decoy_p2])
        is_td = (~is_tt) & (~is_dd)
    else:
        is_dd = df_by_score_desc[decoy_class] == 'DD'
        is_td = df_by_score_desc[decoy_class] == 'TD'
        is_tt = df_by_score_desc[decoy_class] == 'TT'

    n_slices = os.cpu_count()
    if max_slices is not None:
        n_slices = min([n_slices, max_slices])
    slice_size = ceil(len(df)/n_slices)

    with mp.Pool(10) as pool:
        # Calculate sliced cumsum
        tt_cumsum_ares = []
        td_cumsum_ares = []
        dd_cumsum_ares = []
        for i in range(n_slices):
            slice_tt = is_tt.iloc[i*slice_size : (i+1)*slice_size].to_numpy()
            slice_td = is_td.iloc[i*slice_size : (i+1)*slice_size].to_numpy()
            slice_dd = is_dd.iloc[i*slice_size : (i+1)*slice_size].to_numpy()

            tt_cumsum_ares.append(
                pool.apply_async(np.cumsum, (slice_tt,))
            )
            td_cumsum_ares.append(
                pool.apply_async(np.cumsum, (slice_td,))
            )
            dd_cumsum_ares.append(
                pool.apply_async(np.cumsum, (slice_dd,))
            )

        # Wait for cumsum slices
        for x in [tt_cumsum_ares, td_cumsum_ares, dd_cumsum_ares]:
            while np.count_nonzero([not r.ready() for r in x]) > 0:
                sleep(3)

        # Merge slices
        tt_cumsum = None
        for r in tt_cumsum_ares:
            if tt_cumsum is None or len(tt_cumsum) == 0:
                tt_cumsum = r.get()
            else:
                offset = tt_cumsum[-1]
                tt_cumsum = np.append(
                    tt_cumsum,
                    r.get() + offset
                )
        td_cumsum = None
        for r in td_cumsum_ares:
            if td_cumsum is None or len(td_cumsum) == 0:
                td_cumsum = r.get()
            else:
                offset = td_cumsum[-1]
                td_cumsum = np.append(
                    td_cumsum,
                    r.get() + offset
                )
        dd_cumsum = None
        for r in dd_cumsum_ares:
            if dd_cumsum is None or len(dd_cumsum) == 0:
                dd_cumsum = r.get()
            else:
                offset = dd_cumsum[-1]
                dd_cumsum = np.append(
                    dd_cumsum,
                    r.get() + offset
                )

        fdr_raw = (td_cumsum - dd_cumsum) / tt_cumsum
        fdr_raw_flip = np.flip(fdr_raw)

        ares = []
        # Spawn workers
        for i in range(n_slices):
            slice = pd.Series(
                fdr_raw_flip[i*slice_size : (i+1)*slice_size]
            )
            ares.append(
                pool.apply_async(
                    slice.cummin
                )
            )

        # Wait for cummin
        while np.count_nonzero([not r.ready() for r in ares]) > 0:
            sleep(3)

        fdr_flip = None
        # Merge slices
        for r in ares:
            if fdr_flip is None or len(fdr_flip)==0:
                fdr_flip = r.get().to_numpy()
            else:
                fdr_flip = np.append(
                    fdr_flip,
                    np.clip(
                        r.get(),
                        a_min=0,
                        a_max=fdr_flip[-1]
                    )
                )

        df_by_score_desc.loc[:, 'fdr'] = np.flip(fdr_flip)
        return df_by_score_desc.loc[orig_sorting, 'fdr']


def calculate_bi_fdr(df,
                     score_col='match_score',
                     fdr_group_col=None,
                     decoy_class=None,
                     max_slices=None,
                     decoy_p1='decoy_p1',
                     decoy_p2='decoy_p2',
                     str_self='self'):
    original_order = df.index
    if fdr_group_col is None:
        df.loc[:,'fdr_group'] = self_or_between_mp(df)
    else:
        df.loc[:, 'fdr_group'] = df.loc[:, fdr_group_col]
    self = df.loc[:, 'fdr_group'] == str_self

    df_by_index = df.sort_index()
    self_fdr = calculate_fdr(
        df_by_index.loc[self],
        score_col=score_col,
        decoy_class=decoy_class,
        decoy_p1=decoy_p1,
        decoy_p2=decoy_p2,
        max_slices=max_slices
    )
    between_fdr = calculate_fdr(
        df_by_index.loc[~self],
        score_col=score_col,
        decoy_class=decoy_class,
        decoy_p1=decoy_p1,
        decoy_p2=decoy_p2,
        max_slices=max_slices
    )
    return pd.concat([self_fdr, between_fdr]).loc[original_order]

import pandas as pd
from xirescore.bi_fdr import self_or_between_mp, calculate_bi_fdr


def generate(df, options: dict, do_self_between=False, do_fdr=False) -> pd.DataFrame:
    input_cols = options['input']['columns']
    # Generate decoy_class column from decoy_p1 and decoy_p2
    if input_cols['decoy_class'] not in df.columns:
        df[input_cols['decoy_class']] = ''
        df.loc[
            df[input_cols['decoy_p1']] & df[input_cols['decoy_p2']],
            input_cols['decoy_class']
        ] = 'DD'
        df.loc[
            (~df[input_cols['decoy_p1']]) & (~df[input_cols['decoy_p2']]),
            input_cols['decoy_class']
        ] = 'TT'
        df.loc[
            (df[input_cols['decoy_class']] != 'TT') & (df[input_cols['decoy_class']] != 'DD'),
            input_cols['decoy_class']
        ] = 'TD'
    # Generate target column from decoy_class
    if input_cols['target'] not in df.columns:
        df[input_cols['target']] = False
        df.loc[
            df[input_cols['decoy_class']] == 'TT',
            input_cols['target']
        ] = True
    # Calculte self_between from protein_p1, and protein_p2
    if do_self_between and input_cols['self_between'] not in df.columns:
        df[input_cols['self_between']] = self_or_between_mp(
            df,
            col_prot1=input_cols['protein_p1'],
            col_prot2=input_cols['protein_p2'],
        )
    # Calculate fdr from self_between and score
    if do_fdr and input_cols['fdr'] not in df.columns:
        df.loc[
            df[input_cols['top_ranking']],
            input_cols['fdr']
        ] = calculate_bi_fdr(
            df,
            score_col=input_cols['score'],
            decoy_class=input_cols['decoy_class'],
            fdr_group_col=input_cols['self_between'],
        )

    return df
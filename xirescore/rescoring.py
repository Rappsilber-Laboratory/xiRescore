import multiprocess as mp
from math import ceil
import numpy as np
import pandas as pd
from xirescore import async_result_resolving
import scipy
import logging


def rescore(models, df, rescore_col, apply_logit=False, logger=logging.getLogger(__name__)):
    n_procs = int(mp.cpu_count() - 1)
    n_models = len(models)
    n_dataslices = ceil(n_procs/n_models)
    slice_size = len(df) / n_dataslices
    logger.debug(f'Split {len(df)} samples in {n_dataslices} slices of {slice_size}')

    # Slice input data for multiprocessing
    dataslices = [
        df.iloc[
            i*slice_size:(i+1)*slice_size
        ] for i in range(n_dataslices)
    ]

    # Apply each classifier to each data slice
    with mp.Pool(n_procs) as pool:
        async_results = []
        for slice in dataslices:
            for clf in models:
                if "Perceptron" in str(clf):
                    async_results.append(
                        pool.apply_async(
                            clf.decision_function,
                            slice,
                        )
                    )
                elif "tensorflow" not in str(clf):
                    async_results.append(
                        pool.apply_async(
                            clf.predict_proba,
                            slice,
                        )
                    )
                else:
                    async_results.append(
                        pool.apply_async(
                            clf.predict_proba,
                            slice,
                        )
                    )
        rescore_results = np.array(async_result_resolving.resolve(async_results))

    # Apply logit
    if apply_logit:
        rescore_results = scipy.special.logit(rescore_results)

    # Init result DF
    df_rescore = pd.DataFrame(index=df.index)

    # Fill result DF
    for i_m in models:
        n_res = len(async_results)
        rescores_m = rescore_results[i_m:n_res:n_models]
        df_rescore[f'{rescore_col}_{i_m}'] = np.concatenate(rescores_m)

    # Calculate mean score
    df_rescore[rescore_col] = df_rescore.apply(np.mean, axis=1)

    return df_rescore

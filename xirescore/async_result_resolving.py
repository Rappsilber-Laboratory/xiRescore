import logging
import functools
from collections.abc import Iterable
import multiprocess as mp
import time

from tqdm import tqdm


def resolve(async_results, show_pbar=True, pbar_args=dict(), logger=None):
    """
    Resolve asynchronous results from multiprocessing with progress bar.
    For compatibility this function returns anything that is not an asynchronous
    result or and array of these.

    :param async_results: Single async result, list of async results or literals.
    :type async_results: Any

    :returns: Resolved result(s) or unprocessed ``async_results`` if it's a literal.
    :rtype: Any
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logger.getChild(__name__)

    logger.debug("Resolving results")
    if not isinstance(async_results, Iterable):
        logger.debug("Single result")
        if type(async_results) is functools.partial:
            logger.debug("Partial result")
            return async_results()
        if type(async_results) is not mp.pool.AsyncResult:
            logger.debug("Async result")
            return async_results
        return async_results.get()

    logger.debug("List of results")
    pbar = None
    if show_pbar:
        pbar = tqdm(total=len(async_results), dynamic_ncols=True, **pbar_args)
    if type(async_results[0]) is functools.partial:
        logger.debug("Partial results")
        results = []
        for p in async_results:
            results.append(p())
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        return results

    if type(async_results[0]) is not mp.pool.AsyncResult:
        logger.debug("Static results")
        if pbar:
            pbar.close()
        return async_results

    finished = 0
    results = [None for _ in async_results]
    logger.debug("Async results")
    while finished < len(async_results):
        n_finished = 0
        for i, ar in enumerate(async_results):
            if ar.ready():
                n_finished += 1
                if results[i] is None:
                    results[i] = ar.get()

        new_finished = n_finished - finished
        finished = n_finished

        if pbar:
            if new_finished > 0:
                pbar.update(new_finished)
            else:
                pbar.refresh()
        time.sleep(1)

    if pbar:
        pbar.close()
    return results

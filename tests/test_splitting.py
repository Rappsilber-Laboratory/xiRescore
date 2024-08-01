#!/usr/bin/env python

"""Tests for `xirescore` package."""
import pandas as pd
import pytest
import logging
import tempfile
import subprocess
import os
import numpy as np
import networkx as nx
import random
from xirescore.NoOverlapKFold import NoOverlapKFold

def test_no_overlap_splitting():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    random.seed(0)
    np.random.seed(0)
    cluster_size = 300
    n_violations = 3

    # Generate graph with disjunct clusters
    clusters = [
        nx.complete_graph(cluster_size)
        for i in range(1, 6)
    ]
    g_union = nx.union(
        clusters[0],
        clusters[1],
        rename=('0_', '1_')
    )
    for i in range(2, len(clusters)):
        g_union = nx.union(
            g_union,
            clusters[i],
            rename=('', f'{i}_')
        )

    # Add violating edges
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                nodes_i = random.choices(
                    range(cluster_size),
                    k=n_violations,
                )
                nodes_j = random.choices(
                    range(cluster_size),
                    k=n_violations,
                )
                nodes_i = [
                    f'{i}_{n}' for n in nodes_i
                ]
                nodes_j = [
                    f'{j}_{n}' for n in nodes_j
                ]
                edges = zip(
                    nodes_i,
                    nodes_j,
                )
                g_union.add_edges_from(edges)

    # Generate DataFrame from graph
    df = pd.DataFrame(g_union.edges(), columns=['id_p1', 'id_p2'])
    sequences = {
        id_: ''.join(random.choices('ABCDEFGHIJKLMNOP', k=20))
        for id_ in g_union.nodes()
    }
    df['base_sequence_p1'] = df['id_p1'].replace(sequences)
    df['base_sequence_p2'] = df['id_p2'].replace(sequences)
    df['isTT'] = random.choices([True, False], k=len(df))
    kfold = NoOverlapKFold(logger=logger)
    splits = kfold.splits_by_peptides(df)
    n_split_samples = 0
    for (train_idx, test_idx) in splits:
        n_split_samples += len(test_idx)
        groups_p1 = df.loc[test_idx]['id_p1'].str.split('_').apply(lambda x: x[0])
        groups_p2 = df.loc[test_idx]['id_p2'].str.split('_').apply(lambda x: x[0])
        groups = np.unique(groups_p1.to_list() + groups_p2.to_list())
        logger.info(f'groups: {groups}')
        # Assert that less that 10% of training samples were lost
        assert len(train_idx) > 0.9*(len(df)-len(test_idx))
        # Assert that some training samples have been lost
        assert len(train_idx) < (len(df) - len(test_idx))
    assert len(df) == n_split_samples

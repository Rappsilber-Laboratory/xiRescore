import networkx
import pandas as pd
import numpy as np
import logging
import os
import networkx as nx
import re

class NoOverlapKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = 42,
                 pep1_id_col: str = "sequence_p1", pep2_id_col: str = "sequence_p2", logger: logging.Logger = None):
        """
        Constructor for NoOverlapKFold.

        Parameters:
        - n_splits (int, optional): Number of splits. Default is 5.
        - shuffle (bool, optional): Whether to shuffle the data before splitting. Default is False.
        - random_state (int or RandomState, optional): Seed for the random number generator. Default is None.
        - pep1_id_col (str, optional): Column name for PepSeq1. Default is "sequence_p1".
        - pep2_id_col (str, optional): Column name for PepSeq2. Default is "sequence_p2".
        """
        if logger is None:
            logger = logging.getLogger('ximl')
        self.logger = logger.getChild(__name__)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.pep1_id_col = pep1_id_col
        self.pep2_id_col = pep2_id_col

    def splits_by_peptides(self, df: pd.DataFrame, pepseqs: pd.DataFrame = None):
        # Create graph for CSM network
        edges = None
        if pepseqs is not None:
            edges = pepseqs[[self.pep1_id_col, self.pep2_id_col]]
        else:
            edges = df[[self.pep1_id_col, self.pep2_id_col]]
        edges = edges.drop_duplicates()

        # Pepseq grouping
        pepseqs_grouping = pd.concat(
            [
                edges[[self.pep1_id_col]].rename({self.pep1_id_col: 'pepseq'}, axis=1),
                edges[[self.pep2_id_col]].rename({self.pep2_id_col: 'pepseq'}, axis=1),
            ]
        ).drop_duplicates()

        # Create column for converted sequences
        pepseqs_grouping['alt_pepseq'] = pepseqs_grouping['pepseq']

        # Initialize groups
        pepseqs_grouping['group'] = pepseqs_grouping.reset_index().index

        # Merge peptide sequences that only differ by modifications
        self.logger.debug("Disregard modifications")
        pepseqs_grouping = self.regroup_unmodified(pepseqs_grouping)

        # Merge sequences that are contained in other sequences
        #self.logger.debug("Group comprising sequences")
        #pepseqs_grouping = self.regroup_subsequences(pepseqs_grouping)

        n_pepseqs = len(pepseqs_grouping['pepseq'].drop_duplicates())
        n_groups = len(pepseqs_grouping['group'].drop_duplicates())
        self.logger.debug(f"Grouping factor: {n_groups} groups / {n_pepseqs} seqs = {n_groups/n_pepseqs:.2f}")

        # Apply grouping
        self.logger.debug(f"Apply grouping")
        edges_grouped = edges.merge(
            pepseqs_grouping.rename({'group': 'source'}, axis=1),
            left_on=self.pep1_id_col,
            right_on="pepseq",
            validate='many_to_one',
        ).merge(
            pepseqs_grouping.rename({'group': 'target'}, axis=1),
            left_on=self.pep2_id_col,
            right_on="pepseq",
            validate='many_to_one',
        ).drop_duplicates()

        self.logger.debug(f"Construct graph")
        peptide_graph: networkx.Graph = nx.from_pandas_edgelist(edges_grouped)

        # Calculate maximum slice size
        slice_size = int((peptide_graph.number_of_nodes() /self.n_splits))
        self.logger.debug(f"Maximum slice size: {slice_size}")

        # Split into components and communities
        commties = self.recursive_async_fluidc(peptide_graph, max_size=slice_size)
        self.logger.debug("Clustering done.")

        # Map group commties back to peptides
        pepseq_commties = self.group_to_pepseqs(commties, pepseqs_grouping)

        # Recombine into slices
        slices = self.communities_slicing(pepseq_commties, self.n_splits)
        self.logger.debug(f"Slice sizes: {[len(s) for s in slices]}")

        # Convert to pandas dataframe
        slicing_parts = []
        for i, s in enumerate(slices):
            slicing_parts.append(pd.DataFrame(
                [[peptide, i] for peptide in s],
                columns=['kfold_peptide', 'slice']
            ))
        slicing = pd.concat(slicing_parts)

        # Add slice columns for both peptides to dataset
        edges = pd.merge(
            edges,
            slicing,
            left_on=self.pep1_id_col,
            right_on='kfold_peptide',
            how='left',
            validate='many_to_one',
        ).set_axis(edges.index).rename({'slice': 'slice1'}, axis=1)
        
        edges = pd.merge(
            edges,
            slicing,
            left_on=self.pep2_id_col,
            right_on='kfold_peptide',
            how='left',
            validate='many_to_one',
        ).set_axis(edges.index).rename({'slice': 'slice2'}, axis=1)

        # On slice spanning CSMs choose one pseudo randomly
        edges['slice1_hash'] = (edges[self.pep1_id_col] + str(self.random_state)).apply(hash)
        edges['slice2_hash'] = (edges[self.pep2_id_col] + str(self.random_state)).apply(hash)
        edges['decision_hash'] = (edges['slice1_hash']+edges['slice2_hash']).apply(hash)
        edges['slice1_dist'] = np.absolute(edges['slice1_hash'] - edges['decision_hash'])
        edges['slice2_dist'] = np.absolute(edges['slice2_hash'] - edges['decision_hash'])
        edges['slice'] = np.where(
            edges['slice1_dist'] < edges['slice2_dist'],
            edges['slice1'],
            edges['slice2']
        )

        df = df.merge(
            edges[['slice']],
            left_index=True,
            right_index=True,
            how='inner',
            validate='one_to_one',
        )
        splits = [
            (
                df[df['slice'] != i].index,
                df[df['slice'] == i].index
            )
            for i in range(self.n_splits)
        ]

        self.logger.debug("===DEBUG===")
        self.logger.info("Sanity checks")
        testcum = set()
        for train_s, test_s in splits:
            self.logger.debug(f"Train/Test: {len(train_s)}/{len(test_s)}")
            if len(np.intersect1d(train_s, test_s)) > 0:
                self.logger.error("FATAL! Train and test overlapping")
                return None
            testcum.update(list(test_s))
        if len(df) != len(testcum):
            self.logger.error(f"FATAL! Not all training data tested {len(testcum)} of {len(df)}.")
            return None

        return splits

    def _recursive_async_fluidc_comp(self, comp_g: nx.Graph, max_size=1000000, n_communities=2) -> list[set]:
        good_communities = []
        commties = [c for c in nx.community.asyn_fluidc(comp_g, n_communities, seed=self.random_state)]
        commty_counts = [len(c) for c in commties]
        self.logger.debug(f"Community sizes: {commty_counts} = {sum(commty_counts)}")
        for comm in commties:
            if len(comm) <= max_size:
                good_communities += [comm]
            else:
                comm_g = nx.subgraph(comp_g, comm)
                good_communities += self.recursive_async_fluidc(comm_g, max_size=max_size)
        return good_communities

    def recursive_async_fluidc(self, g: nx.Graph, max_size=1000000, n_communities=2) -> list[set]:
        good_communities = []
        comps = [c for c in nx.connected_components(g)]
        self.logger.debug(f"Number of components: {len(comps)}")
        for comp in comps:
            if len(comp) <= max_size:
                good_communities += [comp]
            else:
                comp_g = nx.subgraph(g, comp)
                good_communities += self._recursive_async_fluidc_comp(
                    comp_g,
                    max_size=max_size,
                    n_communities=n_communities
                )
        return good_communities

    def communities_slicing(self, commties, n_slices=2) -> list[set]:
        # Sort communities by size (ascending)
        commty_sizes = np.array([len(c) for c in commties])
        commties = [commties[i] for i in commty_sizes.argsort()]

        slices = [set() for _ in range(n_slices)]

        while len(commties) > 0:
            # Largest community
            commty = commties.pop()
            # Sort slices by size (ascending)
            slice_sizes = np.array([len(s) for s in slices])
            slices = [slices[i] for i in slice_sizes.argsort()]
            # Add to smallest slice
            slices[0].update(commty)

        return slices

    def regroup_unmodified(self, pepseq_grouping: pd.DataFrame) -> pd.DataFrame:
        pepseq_grouping['alt_pepseq'] = pepseq_grouping['alt_pepseq'].apply(
            lambda x: re.sub(r"[^A-Z]", "", x)
        )

        regrouping: pd.DataFrame = pepseq_grouping.groupby('alt_pepseq').min()
        regrouping.rename({'group': 'new_group'}, axis=1, inplace=True)

        pepseq_grouping = pepseq_grouping.merge(
            regrouping[['new_group']],
            left_on='alt_pepseq',
            right_index=True,
            validate='many_to_one',
        )

        pepseq_grouping['group'] = pepseq_grouping['new_group']

        return pepseq_grouping[['pepseq', 'alt_pepseq', 'group']]

    def regroup_subsequences(self, pepseq_grouping: pd.DataFrame, mode='subseq', overlap_factor=0.0) -> pd.DataFrame:
        # Subsequence condition
        def is_subseq(seq1: str, seq2: str):
            if seq1 == seq2:
                return False

            overlapping = False
            if mode == 'subseq':
                overlapping = seq1 in seq2
            elif mode == 'extension':
                overlapping = seq2.startswith(seq1) or seq2.endswith(seq1)
            overlapping_ammount = min([len(seq1), len(seq2)]) / max([len(seq1), len(seq2)])
            return overlapping and (overlapping_ammount >= overlap_factor)

        subseq_graph = nx.Graph()
        subseq_graph.add_nodes_from(pepseq_grouping['alt_pepseq'])

        for s1 in pepseq_grouping['alt_pepseq']:
            for s2 in pepseq_grouping['alt_pepseq']:
                if is_subseq(s1, s2):
                    subseq_graph.add_edge(s1, s2)

        subseq_components = nx.connected_components(subseq_graph)

        regrouping = pd.DataFrame([
            {'alt_pepseq_regroup': list(y_i)} for y_i in subseq_components
        ])

        regrouping['new_group'] = regrouping.reset_index().index
        regrouping = regrouping.explode('alt_pepseq_regroup')

        pepseq_grouping = pepseq_grouping.merge(
            regrouping,
            left_on='alt_pepseq',
            right_on='alt_pepseq_regroup',
            validate='one_to_many',
        )

        pepseq_grouping['group'] = pepseq_grouping['new_group']

        return pepseq_grouping[['pepseq', 'alt_pepseq', 'group']]

    def group_to_pepseqs(self, commties: list[set], pepseqs_grouping: pd.DataFrame):
        pepseq_commties = []
        for c in commties:
            commty_filter = pepseqs_grouping['group'].isin(c)
            pepseqs = pepseqs_grouping.loc[commty_filter, 'pepseq']
            pepseq_commties += [set(pepseqs)]
        return pepseq_commties

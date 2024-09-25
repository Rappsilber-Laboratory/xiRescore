import random
import re
import logging

import networkx
import pandas as pd
import numpy as np
import networkx as nx


class NoOverlapKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = 42,
                 pep1_id_col: str = "base_sequence_p1", pep2_id_col: str = "base_sequence_p2",
                 target_col='isTT',
                 logger: logging.Logger = None):
        """
        Constructor for NoOverlapKFold.

        Parameters:
        - n_splits (int, optional): Number of splits. Default is 5.
        - shuffle (bool, optional): Whether to shuffle the data before splitting. Default is False.
        - random_state (int or RandomState, optional): Seed for the random number generator. Default is None.
        - pep1_id_col (str, optional): Column name for PepSeq1. Default is "base_sequence_p1".
        - pep2_id_col (str, optional): Column name for PepSeq2. Default is "base_sequence_p2".
        """
        if logger is None:
            logger = logging.getLogger('ximl')
        self.logger = logger.getChild(__name__)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.pep1_id_col = pep1_id_col
        self.pep2_id_col = pep2_id_col
        self.target_col = target_col

    def splits_by_peptides(self,
                           df: pd.DataFrame,
                           labels: pd.DataFrame = None,
                           pepseqs: pd.DataFrame = None,
                           n_retries = 10):
        """
        Splits a DataFrame into multiple subsets based on peptide sequences for cross-validation.

        This function creates a graph-based network of peptide sequences and splits the DataFrame
        into a specified number of slices for cross-validation. It considers sequences that are
        only different by modifications and ensures that the same peptide sequences are grouped
        together.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the peptide data.
        labels : pd.DataFrame, optional
            Labeling of the samples. If not provided, the 'df' DataFrame is used.
        pepseqs : pd.DataFrame, optional
            An optional DataFrame containing specific peptide sequences to be used for creating the
            graph. If not provided, the 'df' DataFrame is used.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            A list of tuples, each containing two arrays: the indices of the training set and the
            indices of the test set for each split.

        Notes
        -----
        The function performs the following steps:
        1. Creates a graph for the peptide sequence network.
        2. Groups peptide sequences by disregarding modifications.
        3. Constructs a graph from the grouped sequences.
        4. Calculates the maximum slice size and splits the graph into components.
        5. Maps the grouped sequences back to peptides and recombines them into slices.
        6. Adds slice columns to the original dataset.
        7. Ensures no overlap between train and test sets in each split.

        The method assumes the existence of the following instance attributes:
        - `pep1_id_col`: The column name for the first peptide identifier.
        - `pep2_id_col`: The column name for the second peptide identifier.
        - `n_splits`: The number of splits to be created.
        - `random_state`: A random seed for reproducibility.
        - `logger`: A logger for debugging and information messages.

        Examples
        --------
        >>> splits = obj.splits_by_peptides(df)
        >>> for train_index, test_index in splits:
        >>>     print("TRAIN:", train_index, "TEST:", test_index)
        """
        random.seed(self.random_state)
        df_start = df.copy()
        if labels is None:
            labels = df[[self.target_col]]
        labels_start = labels.copy()
        if pepseqs is None:
            pepseqs = df[[
                self.pep1_id_col,
                self.pep2_id_col,
            ]]
        pepseqs_start = pepseqs.copy()
        for try_run in range(n_retries):
            df = df_start.copy()
            labels = labels_start.copy()
            pepseqs = pepseqs_start.copy()
            seed = random.randint(0, 2**32)

            # Create graph for CSM network
            edges = None
            if pepseqs is not None:
                edges = pepseqs[[self.pep1_id_col, self.pep2_id_col]]
            else:
                edges = df[[self.pep1_id_col, self.pep2_id_col]]

            # Pepseq grouping
            # In the beginning this will just be a enumerated list of all peptides
            pepseqs_grouping = pd.concat(
                [
                    edges[[self.pep1_id_col]].rename({self.pep1_id_col: 'pepseq'}, axis=1),
                    edges[[self.pep2_id_col]].rename({self.pep2_id_col: 'pepseq'}, axis=1),
                ]
            ).drop_duplicates()

            # Create column for converted sequences
            # This was initially intended as the unmodified peptide sequence. Even though we now use
            # the already unmodified base_sequence column, we keep it in here for compatibility.
            pepseqs_grouping['alt_pepseq'] = pepseqs_grouping['pepseq']

            # Initialize groups by assigning each peptide its own group
            pepseqs_grouping['group'] = pepseqs_grouping.reset_index().index

            # Merge peptide sequences that only differ by modifications
            self.logger.debug("Disregard modifications")
            pepseqs_grouping = self.regroup_unmodified(pepseqs_grouping)

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
            slice_size = int((peptide_graph.number_of_nodes() / self.n_splits))
            self.logger.debug(f"Maximum slice size: {slice_size}")

            # Split into components and communities
            commties = self.recursive_async_fluidc(peptide_graph, max_size=slice_size, seed=seed)
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
            splits_clean = [
                None for _ in splits
            ]
            for i, s in enumerate(splits):
                old_size = len(splits[i][0])
                splits_clean[i] = self.cleanup_split(
                    s,
                    pepseqs,
                    seq1col=self.pep1_id_col,
                    seq2col=self.pep2_id_col,
                )
                new_size = len(splits_clean[i][0])
                self.logger.info(
                    f"Split {i} train set clean-up: {new_size}/{old_size}="
                    f"{(100 * new_size / old_size): .2f}% remaining"
                )

            self.logger.debug("===DEBUG===")
            self.logger.info("Sanity checks")
            testcum = set()
            labels_ok = True
            for i_split, (train_s, test_s) in enumerate(splits_clean):
                self.logger.debug(f"Train/Test: {len(train_s):,.0f}/{len(test_s):,.0f}")
                if len(np.intersect1d(train_s, test_s)) > 0:
                    self.logger.fatal("FATAL! Train and test overlapping")
                    return None
                train_label_counts = labels.loc[train_s, self.target_col].value_counts()
                test_label_counts = labels.loc[test_s, self.target_col].value_counts()
                self.logger.debug(f"Slice {i_split} train: {train_label_counts.to_dict()}")
                self.logger.debug(f"Slice {i_split} test: {test_label_counts.to_dict()}")
                if len(train_label_counts) < 2:
                    self.logger.warning(
                        f"Try {try_run+1} of {n_retries} failed: "
                        f"Only one training label in slice {i_split}."
                    )
                    labels_ok = False
                    break
                if len(test_label_counts) < 2:
                    self.logger.warning(
                        f"Try {try_run+1} of {n_retries} failed: "
                        f"Only one test label in slice {i_split}."
                    )
                    labels_ok = False
                    break
                testcum.update(list(test_s))

            if not labels_ok:
                continue

            if len(df) != len(testcum):
                self.logger.error(f"FATAL! Not all training data tested {len(testcum):,.0f} of {len(df):,.0f}.")
                return None

            return splits_clean
        self.logger.critical('Fatal: Could not find a working splitting.')

    def cleanup_split(self, fold, pepseqs, seq1col="sequence_p1", seq2col="sequence_p2"):
        train_index, test_index = fold
        unique_values_test = pd.concat([
            pepseqs.loc[test_index][seq1col],
            pepseqs.loc[test_index][seq2col]
        ]).drop_duplicates()
        conflict_idx = pepseqs.loc[train_index].loc[
            np.isin(pepseqs.loc[train_index][seq1col], unique_values_test) |
            np.isin(pepseqs.loc[train_index][seq2col], unique_values_test)
            ].index
        filtered_train_index = train_index.difference(conflict_idx)
        return filtered_train_index, test_index

    def _recursive_async_fluidc_comp(self, comp_g: nx.Graph, seed, max_size=1000000, n_communities=2) -> list[set]:
        good_communities = []
        commties = [c for c in nx.community.asyn_fluidc(comp_g, n_communities, seed=seed)]
        commty_counts = [len(c) for c in commties]
        self.logger.debug(f"Community sizes: {commty_counts} = {sum(commty_counts)}")
        for comm in commties:
            if len(comm) <= max_size:
                good_communities += [comm]
            else:
                comm_g = nx.subgraph(comp_g, comm)
                good_communities += self.recursive_async_fluidc(comm_g, max_size=max_size, seed=seed)
        return good_communities

    def recursive_async_fluidc(self, g: nx.Graph, seed, max_size=1000000, n_communities=2) -> list[set]:
        good_communities = []
        comps = [c for c in nx.connected_components(g)]
        self.logger.debug(f"Number of components: {len(comps):,.0f}")
        for comp in comps:
            if len(comp) <= max_size:
                good_communities += [comp]
            else:
                comp_g = nx.subgraph(g, comp)
                good_communities += self._recursive_async_fluidc_comp(
                    comp_g,
                    max_size=max_size,
                    n_communities=n_communities,
                    seed=seed
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

    def group_to_pepseqs(self, commties: list[set], pepseqs_grouping: pd.DataFrame):
        pepseq_commties = []
        for c in commties:
            commty_filter = pepseqs_grouping['group'].isin(c)
            pepseqs = pepseqs_grouping.loc[commty_filter, 'pepseq']
            pepseq_commties += [set(pepseqs)]
        return pepseq_commties

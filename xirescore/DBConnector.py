"""
Connector for xiSearch2 databases
"""
import io
import logging
import uuid
import random
from math import ceil
from datetime import datetime

from sqlalchemy import (
    create_engine,
    MetaData,
    select,
    insert,
    and_,
    or_,
    Table as SATable,
    func,
    cast,
    String,
)
import psycopg2
import pandas as pd
import numpy as np

from xirescore.df_serializing import serialize_columns

_TABLES = [
    'resultset',
    'resultsettype',
    'resultmatch',
    'scorename',
    'resultsearch',
    'match',
    'protein',
    'modifiedpeptide',
    'peptideposition',
    'matchedspectrum',
]

_cache_dict = dict()
last_resultset_id_written = None


class DBConnector:
    def __init__(self,
                 hostname: str,
                 port: str,
                 username: str,
                 password: str,
                 database: str,
                 random_seed: int = None,
                 logger: logging.Logger = None):
        if logger is None:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger.getChild(__name__)
        self.engine = create_engine(
            f"postgresql://{username}:{password}@{hostname}:{int(port)}/{database}"
        )
        self.psycopg = psycopg2.connect(
            host=hostname,
            port=port,
            dbname=database,
            user=username,
            password=password,
        )
        self.meta = MetaData()
        self.tables = dict()
        for tbl_name in _TABLES:
            self.tables[tbl_name] = Table(
                tbl_name, self.meta,
                autoload_with=self.engine,
                quote=False
            )
        if random_seed is None:
            random_seed = random.randint(0, 2**32-1)
        self._random_seed = random_seed
        self._random_ctr = 0

    def _random_choice(self, choices):
        rand_state = random.getstate()
        random.seed(self._random_seed + self._random_ctr)
        self._random_ctr += 1
        val = random.choice(choices)
        random.setstate(rand_state)
        return val

    def _cache_load(self, func_name, args):
        global _cache_dict
        key = (func_name, str(args))
        if key in _cache_dict:
            self.logger.debug('Cache hit')
            return _cache_dict[key]
        else:
            self.logger.debug('Cache miss')
            return None

    def _cache_store(self, func_name, args, value):
        global _cache_dict
        key = (func_name, str(args))
        _cache_dict[key] = value

    def _get_search_resset_ids(self, search_ids=[], resultset_ids=[], use_cache=True) -> (set, set):
        self.logger.debug('Fetch resultsearch table')
        with self.engine.connect() as conn:
            ids_query = select(
                cast(self.tables['resultsearch'].c.search_id, String),
                cast(self.tables['resultsearch'].c.resultset_id, String),
            ).where(
                or_(
                    self.tables['resultsearch'].c.search_id.in_(search_ids),
                    self.tables['resultsearch'].c.resultset_id.in_(resultset_ids),
                )
            )
            cache_res = self._cache_load('_get_search_resset_ids', (search_ids, resultset_ids))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_search_resset_ids', (search_ids, resultset_ids), res)
        return {r['search_id'] for r in res}, {r['resultset_id'] for r in res}

    def _get_resultset_df(self, resultset_ids, use_cache=True):
        self.logger.debug('Fetch resultset table')
        with self.engine.connect() as conn:
            resultset_query = select(
                self.tables['resultset'].c.id.label("resultset_id"),
                self.tables['resultset'].c.name.label("resultset_name"),
                self.tables['resultset'].c[
                    'note',
                    'rstype_id',
                    'config',
                    'main_score',
                ],
            ).where(
                self.tables['resultset'].c.id.in_(resultset_ids)
            )
            cache_res = self._cache_load('_get_resultset_df', (resultset_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(resultset_query).mappings().all()
                self._cache_store('_get_resultset_df', (resultset_ids,), res)
        return pd.DataFrame(res)

    def _get_resultmatch_df(self, search_ids, only_top_ranking=False, select_cols: list = None) -> pd.DataFrame:
        self.logger.debug('Fetch resultmatch table')
        with self.engine.connect() as conn:
            # Create column selection
            if select_cols is None:
                # Take all columns if unspecified
                select_cols = self.tables['resultmatch']
            else:
                select_cols = [
                    col
                    for name, col in self.tables['resultmatch'].c.items()
                    if name in select_cols
                ]
            # Construct query
            resultmatch_query = select(
                *select_cols,
            ).where(
                self.tables['resultmatch'].c.search_id.in_(search_ids)
            )
            if only_top_ranking:
                resultmatch_query = resultmatch_query.where(
                    self.tables['resultmatch'].c.top_ranking
                )
            # Execute query
            res = conn.execute(resultmatch_query).mappings().all()
        # Convert to DataFrame
        return pd.DataFrame(res)

    def _get_full_resultmatch_df(self, search_ids,
                                 resultset_ids,
                                 only_top_ranking=False,
                                 only_pairs=False,
                                 spectrum_ids=None,
                                 matchedspec_where=None,
                                 sample: int = None):
        resultmatch_df = self._get_spectrum_result_match_df(
            search_ids=search_ids,
            resultset_ids=resultset_ids,
            only_top_ranking=only_top_ranking,
            only_pairs=only_pairs,
            spectrum_ids=spectrum_ids,
            matchedspec_where=matchedspec_where,
            sample=sample,
        )

        score_names = self._get_score_names(
            resultset_ids=resultset_ids
        )

        # Split scores into columns
        scores_df = pd.DataFrame()
        for rs_id, s_names in score_names.items():
            prefixes = np.array(['feature_']).repeat(len(s_names))
            s_names = np.char.add(prefixes, s_names)
            rs_filter = resultmatch_df['resultset_id'].astype(str) == rs_id
            rs_scores_df = resultmatch_df.loc[rs_filter, ['scores']].apply(
                lambda x: x['scores'],
                result_type='expand',
                axis=1
            )

            rs_scores_df.rename(
                dict(enumerate(s_names)),
                inplace=True,
                axis=1
            )

            rs_scores_df = rs_scores_df.assign(resultset_id=rs_id)

            scores_df = pd.concat([
                scores_df,
                rs_scores_df
            ])

        resultmatch_scores_df = resultmatch_df.merge(
            scores_df,
            left_index=True,
            right_index=True,
            suffixes=('', '_scores'),
            validate='1:1'
        )

        resultmatch_scores_df.drop('scores', inplace=True, axis=1, errors='ignore')

        # Merge with peptide/protein information
        peptide_df = self._get_peptide_protein_df(search_ids)
        peptide1_df = peptide_df.rename(
            {c: f'{c}_p1' for c in peptide_df.columns},
            axis=1
        )
        peptide2_df = peptide_df.rename(
            {c: f'{c}_p2' for c in peptide_df.columns},
            axis=1
        )
        resultmatch_peptides_scores_df = resultmatch_scores_df.merge(
            peptide1_df,
            left_on=['pep1_id'],
            right_on=['peptide_id_p1'],
        ).merge(
            peptide2_df,
            left_on=['pep2_id'],
            right_on=['peptide_id_p2'],
        )

        # Merge with resultset information
        resultset_df = self._get_resultset_df(resultset_ids)

        resultmatch_full_df = resultmatch_peptides_scores_df.merge(
            resultset_df,
            on=['resultset_id'],
            suffixes=('', '_resultset'),
            validate='m:1',
        )

        return resultmatch_full_df

    def _get_match_df(self, search_ids, only_pairs=False) -> pd.DataFrame:
        self.logger.debug('Fetch match table')
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['match'],
            ).where(
                self.tables['match'].c.search_id.in_(search_ids),
            )
            if only_pairs:
                ids_query = ids_query.where(
                    self.tables['match'].c.pep2_id.isnot(None)
                )
            res = conn.execute(ids_query).mappings().all()

        df = pd.DataFrame(res).rename({'id': 'match_id'}, axis=1)
        df['link_score_site1'] = df['link_score_site1'].apply(
            lambda x: ';'.join(np.array(x).astype(str))
        )
        df['link_score_site2'] = df['link_score_site2'].apply(
            lambda x: ';'.join(np.array(x).astype(str))
        )
        df['link_score'] = df['link_score'].apply(
            lambda x: ';'.join(np.array(x).astype(str))
        )
        return df

    def _get_protein_df(self, search_ids, use_cache=True) -> pd.DataFrame:
        self.logger.debug('Fetch protein table')
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['protein'],
            ).where(
                self.tables['protein'].c.search_id.in_(search_ids),
            )
            cache_res = self._cache_load('_get_protein_df', (search_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_protein_df', (search_ids,), res)
        return pd.DataFrame(res).rename({
            'id': 'protein_id',
            'name': 'protein_name',
            'sequence': 'protein_sequence',
        }, axis=1)

    def _get_peptide_df(self, search_ids, use_cache=True) -> pd.DataFrame:
        self.logger.debug('Fetch modifiedpeptide table')
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['modifiedpeptide'],
            ).where(
                self.tables['modifiedpeptide'].c.search_id.in_(search_ids),
            )
            cache_res = self._cache_load('_get_peptide_df', (search_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_peptide_df', (search_ids,), res)
        df = pd.DataFrame(res)
        df = df.rename({'id': 'peptide_id'}, axis=1)

        # Convert array columns to string
        df['modification_ids'] = df['modification_ids'].apply(
            lambda x: ';'.join(np.array(x).astype(str))
        )
        df['modification_position'] = df['modification_position'].apply(
            lambda x: ';'.join(np.array(x).astype(str))
        )
        return df

    def _get_peptideposition_df(self, search_ids, use_cache=True) -> pd.DataFrame:
        self.logger.debug('Fetch peptideposition table')  # FIXME took >20mins
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['peptideposition'],
            ).where(
                self.tables['peptideposition'].c.search_id.in_(search_ids),
            )
            cache_res = self._cache_load('_get_peptideposition_df', (search_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_peptideposition_df', (search_ids,), res)
        return pd.DataFrame(res)

    def _get_matchedspectrum_df(self, search_ids, use_cache=True) -> pd.DataFrame:
        self.logger.debug('Fetch matchedspectrum table')
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['matchedspectrum'],
            ).where(
                self.tables['matchedspectrum'].c.search_id.in_(search_ids)
            )
            cache_res = self._cache_load('_get_matchedspectrum_df', (search_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_matchedspectrum_df', (search_ids,), res)
        return pd.DataFrame(res)

    def _get_spectrum_result_match_df(self,
                                      search_ids: list,
                                      resultset_ids: list,
                                      only_top_ranking=False,
                                      only_pairs=False,
                                      matchedspec_where=None,
                                      spectrum_ids: list = None,
                                      sample: int = None,
                                      rs_oversample: int = 3):
        self.logger.debug('Fetch matchedspectrum, match, resultmatch tables joined')
        with self.engine.connect() as conn:
            # Spectrum pre-subquery
            spectrum_preq = select(
                func.aggregate_strings(
                    cast(self.tables['matchedspectrum'].c.spectrum_id, String),
                    ';'
                ).label('spectrum_id'),
                self.tables['matchedspectrum'].c.search_id,
                self.tables['matchedspectrum'].c.match_id,
            ).where(
                self.tables['matchedspectrum'].c.search_id.in_(search_ids)
            )

            # Add optional where
            if matchedspec_where is not None:
                spectrum_preq = spectrum_preq.where(
                    matchedspec_where
                )

            # Select spectra IDs
            if spectrum_ids is not None:
                spectrum_preq = spectrum_preq.where(
                    spectrum_preq.c.spectrum_id.in_(spectrum_ids)
                )

            # Add grouping
            spectrum_agg = spectrum_preq.group_by(
                self.tables['matchedspectrum'].c.match_id,
                self.tables['matchedspectrum'].c.search_id,
            )

            spectrum_subq = spectrum_agg.alias('spectrum_subq')

            # Match subquery
            match_subq = select(
                *[
                    c for name, c in self.tables['match'].c.items() if name != 'id'
                ],
                self.tables['match'].c.id.label('match_id')
            ).where(
                self.tables['match'].c.search_id.in_(search_ids)
            )
            if only_pairs:
                match_subq = match_subq.where(
                    self.tables['match'].c.pep2_id.isnot(None)
                )
            match_subq = match_subq.alias('match_subq')

            # Resultmatch subquery
            resultmatch_subq = select(
                self.tables['resultmatch'],
            ).where(
                and_(
                    self.tables['resultmatch'].c.search_id.in_(search_ids),
                    self.tables['resultmatch'].c.resultset_id.in_(resultset_ids),
                )
            )
            if only_top_ranking:
                resultmatch_subq = resultmatch_subq.where(
                    self.tables['resultmatch'].c.top_ranking
                )
            if sample is not None:
                resultmatch_subq = resultmatch_subq.order_by(
                    func.random()
                ).limit(sample*rs_oversample)  # Oversample for linear filter
            resultmatch_subq = resultmatch_subq.alias('resultmatch_subq')

            # Join subqueries
            joined_query = select(
                spectrum_subq,
                match_subq,
                resultmatch_subq,
            ).join_from(
                spectrum_subq,
                match_subq,
                and_(
                    spectrum_subq.c.match_id == match_subq.c.match_id,
                    spectrum_subq.c.search_id == match_subq.c.search_id,
                )
            ).join_from(
                match_subq,
                resultmatch_subq,
                and_(
                    match_subq.c.match_id == resultmatch_subq.c.match_id,
                    match_subq.c.search_id == resultmatch_subq.c.search_id,
                )
            )
            if sample is not None:
                joined_query = joined_query.limit(sample)
            res = conn.execute(joined_query).mappings().all()
        return pd.DataFrame(res)

    def _uuid_slice_filter(self, query, column, slice_idx, n_slices):
        slice_size = ceil((2 ** 128) / n_slices)
        spec_id_min = uuid.UUID(
            hex(slice_size * slice_idx)[2:].zfill(32)
        )
        spec_id_max = uuid.UUID(
            hex(slice_size * (slice_idx + 1))[2:].zfill(32)
        )
        query = query.where(
            column >= spec_id_min
        )
        if slice_idx + 1 < n_slices:
            query = query.where(
                column < spec_id_max
            )
        else:
            query = query.where(
                column <= 'f' * 32
            )
        return query

    def _get_score_names(self, resultset_ids, use_cache=True) -> dict:
        self.logger.debug('Fetch scorename table')
        with self.engine.connect() as conn:
            ids_query = select(
                self.tables['scorename'].c.resultset_id,
                self.tables['scorename'].c.name,
            ).where(
                self.tables['scorename'].c.resultset_id.in_(resultset_ids)
            ).order_by(
                self.tables['scorename'].c.resultset_id,
                self.tables['scorename'].c.score_id
            )
            cache_res = self._cache_load('_get_score_names', (resultset_ids,))
            if use_cache and cache_res is not None:
                res = cache_res
            else:
                res = conn.execute(ids_query).mappings().all()
                self._cache_store('_get_score_names', (resultset_ids,), res)
        return {
            rs_id: [
                x['name'] for x in res if str(x['resultset_id']) == rs_id
            ] for rs_id in resultset_ids
        }

    def _get_peptide_protein_df(self, search_ids):
        protein_df = self._get_protein_df(search_ids=search_ids)
        peptideposition_df = self._get_peptideposition_df(
            search_ids=search_ids,
        )
        peptide_df = self._get_peptide_df(
            search_ids=search_ids
        )

        full_df = peptide_df.merge(
            peptideposition_df,
            left_on=['peptide_id', 'search_id'],
            right_on=['mod_pep_id', 'search_id'],
            suffixes=('', '_peppos'),
        ).merge(
            protein_df,
            on=['protein_id'],
            suffixes=('', '_protein'),
        )

        full_df = full_df.groupby(peptide_df.columns.to_list()).agg(
            protein=pd.NamedAgg('protein_name', ';'.join),
            pep_pos=pd.NamedAgg('start', lambda x: ';'.join(np.array(x).astype(str))),
        ).reset_index()

        return full_df

    def read_resultsets(self,
                        resultset_ids: list[str],
                        only_top_ranking=False,
                        only_pairs=False,
                        spectrum_ids=None,
                        sample: int = None) -> pd.DataFrame:
        search_ids, resultset_ids = self._get_search_resset_ids(resultset_ids=resultset_ids)
        df = self._get_full_resultmatch_df(
            search_ids=search_ids,
            resultset_ids=resultset_ids,
            only_top_ranking=only_top_ranking,
            only_pairs=only_pairs,
            spectrum_ids=spectrum_ids,
            sample=sample
        )
        return serialize_columns(df)

    def read_spectra_range(self,
                           resultset_ids: list[str],
                           spectra_from,
                           spectra_to,
                           only_top_ranking=False,
                           only_pairs=False,
                           sample: int = None) -> pd.DataFrame:
        search_ids, resultset_ids = self._get_search_resset_ids(resultset_ids=resultset_ids)
        # Construct optional filter
        matchedspec_where = (
            (self.tables['matchedspectrum'].c.spectrum_id >= spectra_from) &
            (self.tables['matchedspectrum'].c.spectrum_id <= spectra_to)
        )
        # Get filtered resultsets
        df = self._get_full_resultmatch_df(
            search_ids=search_ids,
            resultset_ids=resultset_ids,
            only_top_ranking=only_top_ranking,
            only_pairs=only_pairs,
            matchedspec_where=matchedspec_where,
            sample=sample
        )
        return serialize_columns(df)

    def _get_tailing_uuid(self, len_timestamp=10, len_leading_f=4):
        timestamp = hex(
            int(
                datetime.now().timestamp()
            )
        )[2:].zfill(len_timestamp)
        len_rand = 32 - len_timestamp - len_leading_f
        random_hex = ''.join(
            self._random_choice('0123456789abcdef')
            for _ in range(len_rand)
        )
        resultset_id = ('f' * len_leading_f) + timestamp + random_hex
        return resultset_id

    def create_resultset(self, resultset_name, score_names, main_score, config, search_ids, rs_type='xiRescore'):
        global last_resultset_id_written
        tables = self._get_tables()
        resultset_id = self._get_tailing_uuid()
        last_resultset_id_written = resultset_id
        rstype_id = self._get_rstype_id(rs_type)

        main_score_idx = score_names.index(main_score)

        rs_query = insert(
            tables['resultset']
        ).values({
            'id': resultset_id,
            'name': resultset_name,
            'main_score': main_score_idx,
            'rstype_id': rstype_id,
            'config': config,
        })

        sn_query = insert(
            tables['scorename']
        ).values([
            {
                'resultset_id': resultset_id,
                'score_id': i,
                'name': name,
                'primary_score': name == main_score,
                'higher_is_better': True
            } for i, name in enumerate(score_names)
        ])

        rsrch_query = insert(tables['resultsearch']).values([
            {
                'search_id': search_id,
                'resultset_id': resultset_id,
            } for search_id in search_ids
        ])

        with self.engine.connect() as conn:
            self.logger.debug("Create resultset")
            conn.execute(rs_query)
            self.logger.debug("Create scorenames")
            conn.execute(sn_query)
            self.logger.debug("Create resultsearches")
            conn.execute(rsrch_query)
            conn.commit()

        return resultset_id

    def write_resultmatches(self, df, resultset_id, feature_cols, top_ranking_col='top_ranking'):
        rm_columns = [
            c.name for c in self.meta.tables['resultmatch'].c
        ]
        rm_df_columns = [c for c in rm_columns if c != 'scores']
        rm_df = df.loc[:, rm_df_columns].copy()
        rm_df['top_ranking'] = df[top_ranking_col]
        rm_df['scores'] = df[
            feature_cols
        ].astype(float).apply(
            np.array,
            result_type='reduce',
            axis=1
        )
        rm_df['link_score_site1'] = rm_df['link_score_site1'].str.split(';')
        rm_df['link_score_site2'] = rm_df['link_score_site2'].str.split(';')
        rm_df['link_score'] = rm_df['link_score'].str.split(';')
        for c in ['scores', 'link_score_site1', 'link_score_site2', 'link_score']:
            rm_df[c] = rm_df[c].apply(
                lambda x:
                '{' +
                ','.join(np.array(x).astype(str)) +
                '}'
            )
        rm_df = rm_df.loc[:, rm_columns]

        rm_df['resultset_id'] = resultset_id

        with self.psycopg as psycopg:
            self.logger.debug("Insert resultmatches")
            f = io.StringIO(rm_df.to_csv(index=False))
            with psycopg.cursor() as cursor:
                cursor.copy_expert("COPY resultmatch FROM STDIN (FORMAT CSV, HEADER true)", f)

    def _get_rstype_id(self, name='xiRescore', create=True):
        self.logger.debug('Fetch resultsettype table')
        with self.engine.connect() as conn:
            tables = self._get_tables()
            rstype_id_query = select(
                tables['resultsettype'].c.id
            ).where(
                tables['resultsettype'].c.name == name
            )
            id_res = conn.execute(rstype_id_query).mappings().all()
        if len(id_res) == 0 and create:
            return self._create_rstype(name)
        return id_res[0]['id']

    def _create_rstype(self, name):
        self.logger.debug(f'Create rstype {name}')
        tables = self._get_tables()
        with self.engine.connect() as conn:
            rstype_query = insert(
                tables['resultsettype']
            ).values({
                'name': name,
                'id': select(
                    func.max(tables['resultsettype'].c.id)+1
                ).scalar_subquery(),
            })
            conn.execute(rstype_query)
            conn.commit()
        return self._get_rstype_id(name, create=False)

    def _get_tables(self):
        tables = dict()
        for tbl_name in _TABLES:
            tables[tbl_name] = Table(
                tbl_name, self.meta,
                autoload_with=self.engine,
                quote=False
            )
        return tables

    def read_spectrum_ids(self, resultset_ids):
        search_ids, resultset_ids = self._get_search_resset_ids(resultset_ids=resultset_ids)

        self.logger.debug('Fetch matchedspectrum table')
        with self.engine.connect() as conn:
            spectrum_query = select(
                func.aggregate_strings(
                    cast(self.tables['matchedspectrum'].c.spectrum_id, String),
                    ';'
                ).label('spectrum_id')
            ).where(
                self.tables['matchedspectrum'].c.search_id.in_(search_ids)
            ).group_by(
                self.tables['matchedspectrum'].c.match_id,
                self.tables['matchedspectrum'].c.search_id,
            )
            res = conn.execute(spectrum_query).mappings().all()
        df = pd.DataFrame(res)
        return serialize_columns(df)

    def get_last_resultset_id_written(self):
        global last_resultset_id_written
        return last_resultset_id_written

    def close(self):
        self.engine.dispose()
        self.psycopg.close()


def Table(name: str, *args, **kw):
    """
    Return an SQLAlchemy table but that uses the lower case table name.
    This is a workaround for the "quote=False" argument not working properly for the postgresql
    dialect in SQLAlchemy.

    :param name: Name of the table. Will be forwarded as lower case string.
    :type name: str
    :param *args: Arguments to be passed to the SQLAlchemy Table constructor.
    :param *kw: Keyword arguments to be passed to the SQLAlchemy Table constructor.

    :returns: SQLAlchemy Table
    :rtype: Table
    """
    return SATable(name.lower(), *args, **kw)

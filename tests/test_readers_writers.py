import pytest

import xirescore.readers as readers

@pytest.mark.db
def test_db_input():
    # docker run --rm --name xirescore-pytest-postgres -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test -e POSTGRES_DB=xisearch2 -p 5432:5432 -v ./tests/fixtures/db:/docker-entrypoint-initdb.d/ postgres
    # Read top_ranking sample
    input_db = 'xi2resultsets://test:test@localhost:5432/xisearch2/' \
               'fdbe9e59-2baa-44cb-b8cb-e8b7a590e136'
    df = readers.read_sample(input_db, sample=100, only_top_ranking=True)
    assert len(df) == 100
    del df

    # Read spectra IDs
    spectra_list = readers.read_spectra_ids(input_db)
    spectra_list.sort()
    assert spectra_list.shape[0] > 0

    # Read spectra range
    df = readers.read_spectra_range(
        input_db,
        spectra_from=spectra_list[2],
        spectra_to=spectra_list[10],
    )

    assert len(df) > 0

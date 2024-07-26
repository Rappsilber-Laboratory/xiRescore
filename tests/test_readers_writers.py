import xirescore.readers as readers


def test_db_input():
    # docker run --rm --name xirescore-pytest-postgres -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test -e POSTGRES_DB=xisearch2 -p 5432:5432 -v ./tests/fixtures/db:/docker-entrypoint-initdb.d/ postgres
    # Read top_ranking sample
    input_db = 'xi2resultsets://test:test@localhost:5432/xisearch2/' \
               'fdbe9e59-2baa-44cb-b8cb-e8b7a590e136'
    df = readers.read_top_sample(input_db, sample=100)
    assert len(df) == 100
    del df

    # Read spectra IDs
    spectra_list = readers.read_spectra_ids(input_db)
    assert spectra_list.shape[0] > 0

    # Read spectra range
    df = readers.read_spectra_range(
        input_db,
        spectra_from=spectra_list[2],
        spectra_to=spectra_list[10],
    )

    assert len(df) > 0


def test_parquet_input_output():

    raise Exception("Not implemented yet.")


def test_csv_input():
    raise Exception("Not implemented yet.")


def test_df_input():
    raise Exception("Not implemented yet.")

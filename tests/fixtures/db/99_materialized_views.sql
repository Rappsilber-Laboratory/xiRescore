CREATE MATERIALIZED VIEW matchedspectrum_agg AS
    SELECT match_id,
           search_id,
           ARRAY_AGG(matchedspectrum_type) AS matchedspectrum_type,
           ARRAY_AGG(spectrum_id) AS spectrum_id
    FROM matchedspectrum
    GROUP BY match_id, search_id;

CREATE UNIQUE INDEX matchedspectrum_agg_unique_idx ON matchedspectrum_agg (search_id, spectrum_id, match_id);

CREATE MATERIALIZED VIEW peptide_protein_agg AS
    SELECT
        modifiedpeptide.base_sequence AS base_sequence,
        modifiedpeptide.sequence AS sequence,
        modifiedpeptide.mass AS mass,
        modifiedpeptide.length AS length,
        modifiedpeptide.modification_ids AS modification_ids,
        modifiedpeptide.modification_position AS modification_position,
        modifiedpeptide.is_decoy AS is_decoy,
        protein_agg.mod_pep_id AS mod_pep_id,
        protein_agg.search_id AS search_id,
        protein_agg.peptide_start AS peptide_start,
        protein_agg.protein_id AS protein_id,
        protein_agg.protein_accession AS protein_accession,
        protein_agg.protein_name AS protein,
        protein_agg.gen_name AS gen_name,
        protein_agg.protein_description AS protein_description,
        protein_agg.protein_sequence AS protein_sequence,
        protein_agg.protein_full_header AS protein_full_header
    FROM
        modifiedpeptide
    JOIN (
        SELECT
            peptideposition.mod_pep_id AS mod_pep_id,
            peptideposition.search_id AS search_id,
            ARRAY_AGG(peptideposition.start) AS peptide_start,
            ARRAY_AGG(protein.id) AS protein_id,
            ARRAY_AGG(protein.accession) AS protein_accession,
            ARRAY_AGG(protein.name) AS protein_name,
            ARRAY_AGG(protein.gen_name) AS gen_name,
            ARRAY_AGG(protein.description) AS protein_description,
            ARRAY_AGG(protein.sequence) AS protein_sequence,
            ARRAY_AGG(protein.full_header) AS protein_full_header
        FROM protein
        JOIN peptideposition
        ON peptideposition.search_id = protein.search_id AND peptideposition.protein_id = protein.id
        GROUP BY
            peptideposition.mod_pep_id, peptideposition.search_id
    ) AS protein_agg
    ON protein_agg.search_id = modifiedpeptide.search_id AND protein_agg.mod_pep_id = modifiedpeptide.id;

CREATE UNIQUE INDEX protein_agg_unique_idx ON peptide_protein_agg (search_id, mod_pep_id);

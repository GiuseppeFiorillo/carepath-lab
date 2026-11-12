# Phase 1 learning exercises

These are the learner's earlier exercises, preserved separately from the four
operational commands in the [root README](../../README.md). They are optional and are not required
for full-source ingestion, audit or future model training. Shared CSV validation
now lives in `carepath_lab/validation.py`.

Run commands from the **project root**, using `-m` so the shared package can be
imported. Dependencies and source CSVs are described in the root README.

## Original CSV aggregation exercise

```bash
.venv/bin/python -m examples.phase_1.inspect_csv
```

This preserves the learner-written loops for target/age frequencies, admission
types, distinct patients, repeated encounters and encounter-ID uniqueness. It
reads the source and prints aggregate values. This is the original `read_header`
learning exercise, now outside the operational workflow. The separate
assistant-written inspector was removed as a duplicate.

## Five-column staging and SQL examples

The SQL files here operate on the original `encounters_staging` teaching table,
not on the complete `encounters_raw` table or the selected analysis cohort.
Existing tables/data are not changed by moving these files. Skip creation/loading
if the corresponding objects already exist and are populated.

For a new exercise setup, with the project database/server already prepared:

```bash
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/001_create_encounters_staging.sql
.venv/bin/python -m examples.phase_1.load_staging
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/003_readmission_by_age.sql
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/005_readmission_by_age_cte.sql
```

SQL `002`-`005` cover aggregation, `HAVING` and CTEs. To exercise the mapping join,
create/load the separate eight-row mapping table once:

```bash
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/006_create_admission_type_mapping.sql
.venv/bin/python -m examples.phase_1.load_admission_type_mapping
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/007_encounters_by_admission_type.sql
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f examples/phase_1/sql/008_unmapped_admission_types.sql
```

Both loaders reject populated destinations and preserve source files. Run only
one loader at a time; the empty-table check is not a concurrency lock.

## Transaction rollback demonstration

```bash
.venv/bin/python -m examples.phase_1.transaction_rollback
```

This creates a temporary table, inserts a synthetic value, deliberately raises
an exception and checks that zero rows remain after rollback. Persistent source
tables are not modified. It demonstrates a transaction, not an end-to-end test
of the production loader.

# Business Entity Resolution — Pipeline

Reproduction instructions for this submission's code. See
`docs/dataset_description.md` (repo root) for the data profiling that
motivated the choices below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Pipeline stages

### 1. Preprocessing

Normalizes `business_name` / `business_address` for all six source files
(`train`/`test` × `source1`/`source2`/`source3`) and writes one processed TSV
per input file.

```bash
python code/business_entity_resolution/src/preprocess.py \
    --dataset-dir dataset \
    --output-dir data/processed
```

What it does (implemented in `src/text_normalize.py`):

- Strips synthetic diacritic noise from Latin-script text (`Nétwork` →
  `Network`), gated on a script check so genuine non-Latin script
  (Devanagari/Tamil/Gujarati — see dataset doc §4.1) passes through
  untouched instead of being corrupted.
- Lowercases, tokenizes, and canonicalizes legal-suffix variants to a short
  form (`Corporation`/`Corp` → `corp`, `Private`/`Pvt` → `pvt`, ...) so
  token-based blocking doesn't fragment on suffix spelling.
- Expands common street-type abbreviations in addresses (`St`/`Rd`/`Ave` →
  `street`/`road`/`avenue`, ...).
- Extracts a best-effort postal code (`postal_code` column) — sparse by
  construction of the source data (~10% US, ~0% India), so downstream
  blocking should treat it as a bonus exact-match signal, never a required
  key.
- Emits a `name_is_latin` flag per row, since name-based blocking channels
  have no signal on non-Latin-script names and should defer to address-based
  channels for those rows.

Output columns: `entity_id, country, business_name, business_address,
name_norm, addr_norm, postal_code, name_is_latin`. Raw `business_name` /
`business_address` are kept alongside the normalized fields so later stages
(e.g. an embedding-based channel) can still use the original text.

Runtime: ~150k rows/sec single-threaded (measured on this dataset); the full
~24M rows across all six files takes a few minutes.

### 2. Candidate generation (blocking)

*Not yet implemented.*

### 3. Matching model

*Not yet implemented.*

## Tests

```bash
python code/business_entity_resolution/tests/test_text_normalize.py
```

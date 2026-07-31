# Legacy audit

## Scope and safety

This document records research performed before the version 2 implementation.
The canonical repository at `links1234/lol-esports-win-predictor` is the only implementation source of truth.
The repositories and checkout listed below were inspected as read-only research material.
No legacy implementation is merged into version 2.

The local legacy checkout at `/home/links/projects/lolchampselect_gemini/lol-draft-predictor` was inspected without running checkout, reset, clean, stash, fetch, or any write operation inside that checkout.
It was on local branch `tfmodel` at commit `4e7e1c373b66acc6206af3a1fc50c73d040ae60e` and contained tracked edits, tracked deletions, and untracked source, data, models, and MLflow state.

A credential-safe recovery snapshot was created outside both repositories at `/home/links/recovery/lol-draft-predictor-local-tfmodel-4e7e1c3-20260730`.
The snapshot preserves the tracked source and configuration patch plus selected untracked source and configuration files.
Generated models, datasets, database files, MLflow state, images, environment files, and credentials were deliberately excluded.
The snapshot records its base commit, contents, recovery instructions, and SHA-256 checksums.

## Environment-file finding

The public `links1234/lol-draft-predictor-Links` history contains a committed `.env` file.
The value is the placeholder `YOUR_API_KEY_HERE`, not an active credential.
The file still demonstrates unsafe repository hygiene and is excluded from the proposed tree.
Any history rewrite, repository visibility change, or remote cleanup requires explicit owner approval and is outside this rebuild.

Version 2 never reads or copies that credential.
Environment files, raw credentials, and generated local state are ignored.
The only environment template is `.env.example`, which contains placeholders.

## Canonical repository

Repository: `https://github.com/links1234/lol-esports-win-predictor`

Observed main commit: `4e1842a5393f72d03cbb99526cda264e8063b330`

### What it attempted

The canonical repository provided a small Python package with a wrapper around the unofficial LoL Esports persisted-query API.
It could list leagues, retrieve an event, and extract completed champion picks from a game draft.
It included two small tests and example scripts.

### Useful concepts

- Keep external API access behind a dedicated client.
- Represent a draft as explicit blue-side and red-side picks.
- Make API access configurable through environment variables.
- Test response parsing without making live network calls.

### Problems

- The package used `setup.py` and a top-level package instead of a modern `pyproject.toml` and `src` layout.
- HTTP calls had no explicit timeout, retry policy, response schema validation, or durable ingestion boundary.
- The draft parser did not capture teams, rosters, bans, tournament, patch, or match timestamp.
- There was no data store, feature pipeline, model, temporal evaluation, artifact contract, or end-to-end workflow.
- The README referenced an `.env.example` and ignore rules that were not present at the observed main commit.

### Version 2 decision

The API boundary and testable parsing idea are preserved.
The package, schemas, storage, modeling, and command-line workflows are rebuilt from first principles.
Live API ingestion is not required for the first fixture-backed vertical slice and remains a later adapter.

## Links legacy repository

Repository: `https://github.com/links1234/lol-draft-predictor-Links`

The safe source review used Git metadata and source paths already present in the protected local legacy clone.
The committed `.env` file was not opened.

### What it attempted

The first iteration transformed Oracle's Elixir player-level CSV files into one game per row with teams, players, picks, bans, patch, league, and winner.
It one-hot encoded teams, leagues, patches, picks, and bans, then trained XGBoost classifiers.
Later branches added team form, player-champion and champion statistics, TrueSkill-like ratings, TensorFlow embeddings, and a prediction script.

### Useful concepts

- Oracle's Elixir data is a plausible future public-data ingestion source.
- A game-level record should retain teams, players, picks, bans, patch, league, and result.
- Team ratings must be captured before the result updates the rating state.
- Team, champion, and player-champion history can add useful draft context.
- Both all-region and league-specific reporting are valuable.

### What failed

- The public repository committed a placeholder `.env` file instead of an example template.
- Models and training logs were committed repeatedly.
- List-like CSV values were parsed with `eval()`.
- XGBoost evaluation used random `train_test_split`.
- One-hot vocabularies were fitted before the split.
- The purported same-patch leakage fix subtracted only the target game from full-patch totals.
  It still used games later in the same patch.
- Rows were ordered by year and patch instead of exact match timestamp.
- Neural experiments also used random splitting and accuracy-focused evaluation.
- Model selection relied heavily on accuracy and classification reports.
- Artifacts did not contain a reliable data cutoff, input schema, feature version, or dependency manifest.
- Hard-coded feature dictionaries made live prediction drift away from training.

The legacy logs include accuracy jumps as high as roughly 0.96 on a random holdout.
Those numbers are invalid evidence because they coincide with the leaking feature design and random evaluation.
They must not be used as a benchmark for version 2.

### Version 2 decision

The game-level draft schema, point-in-time rating concept, and public-data direction are preserved.
The legacy feature code, encodings, serialized models, environment file, and reported performance are discarded.

## Maurice repository and protected local checkout

Repository: `https://github.com/MauriceAK/lol-draft-predictor`

Protected local checkout: `/home/links/projects/lolchampselect_gemini/lol-draft-predictor`

### What it attempted

This lineage introduced DuckDB, explicit SQL transformations, team form tables, champion statistics, player-champion statistics, Elo or TrueSkill experiments, a shared-tower Siamese network, configuration-driven runs, MLflow tracking, and live and batch prediction scripts.
The protected checkout also contained an unfinished transition from TrueSkill to a Riot-inspired Elo power score.

### Useful concepts

- DuckDB is a good local analytical store for reproducible, inspectable data work.
- Configuration files should define experiments.
- Experiment runs should record their config, metrics, data cutoff, and artifacts.
- Training and prediction should share feature-generation code.
- Live and batch prediction are both product requirements.
- Shared representations for equivalent team inputs are conceptually sound if a neural model is eventually justified.

### What failed

- Champion meta tables aggregated the full current patch.
- Player and team summaries were patch aggregates rather than exact timestamp-bounded state.
- The unfinished Elo path computed final ratings and reused them for historical rows.
- The training script used random `train_test_split`.
- Equivalent team inputs were transformed with separate scalers.
- Live prediction rebuilt feature maps from the latest database, so it did not reproduce the training feature state.
- Team names depended on a manually generated mapping file.
- The tournament date loader executed a data file with `exec()`.
- Batch evaluation emphasized accuracy and used feature magnitude as a misleading proxy for contribution.
- Raw CSVs, DuckDB files, Keras models, pickles, MLflow state, and virtual-environment assumptions lived in or beside the repository.
- The package had no meaningful automated leakage, integration, or end-to-end tests.
- Deep learning was attempted before reliable probabilistic baselines existed.

The MLflow state in the protected checkout recorded validation accuracy near 0.72 and validation loss near 0.545 for one full feature run, with a power-score-only run near 0.61 accuracy and 0.658 loss.
These were random, leakage-prone validation runs and are retained only as historical context.
They are not comparable to version 2 chronological results.

### Version 2 decision

DuckDB, explicit transformations, configuration-driven experiments, reproducible run records, and batch/live interfaces are retained.
The SQL and Python implementations are not copied.
The Siamese model is deferred until temporal backtesting proves that simpler models leave meaningful log-loss improvement available.

## Reuse policy

Version 2 may reuse product concepts, field names that match public datasets, and general algorithms such as pre-match Elo.
It must not reuse credentials, `.env` files, raw or private datasets, databases, model binaries, pickle files, MLflow directories, virtual environments, hard-coded mappings, generated logs, or legacy performance claims.
Legacy source may inform tests and failure cases, but implementation is written anew against the architecture contract.

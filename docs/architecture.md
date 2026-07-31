# Architecture

## Objective

The system estimates calibrated blue-side and red-side win probabilities for a professional League of Legends game after the draft is known.
The broadcast may already describe the game as live, but the estimate is still a post-draft, pre-game-information estimate.
The model must not consume kills, gold, objectives, game time, items, levels, map state, or any event from after the target game began.
The primary engineering constraint is point-in-time correctness.
Every feature for a target match must be computable from information that existed strictly before that match began.

## Data flow

```text
pinned source snapshots
        |
        v
strict source adapters and quarantine
        |
        v
      DuckDB
        |
        v
point-in-time state replay
        |
        +--> historical feature table with provenance
        |
        v
chronological train / validation / test split
        |
        v
base rate, Elo, logistic, histogram-tree, CatBoost, hierarchical state, and fixed ensembles
        |
        v
validation log-loss selection
        |
        +--> pre-holdout refit and chronological holdout report
        |
        v
preregistered promotion gates and simpler fallback
        |
        v
versioned model artifact plus feature-state snapshot
        |
        v
separate batch or live prediction process
```

Ingestion, validation, feature generation, training, evaluation, artifact loading, and prediction are separate modules.
The command-line interface composes those modules without creating a second implementation path.

The screenshot path has its own trust boundary:

```text
broadcast screenshot
        |
        v
versioned overlay profile and template catalog
        |
        v
field candidates with confidence and runner-up evidence
        |
        +--> review or correction when confidence is insufficient
        |
        v
validated pre-game DraftRequest
        |
        v
the same artifact-backed predictor used by JSON and batch prediction
```

The screenshot parser does not predict a winner directly from pixels.
It extracts auditable draft facts, and the normal feature pipeline predicts from the confirmed facts.
This separation makes parser errors measurable and prevents an image model from silently learning scoreboard or broadcast-result cues.

The current public adapters are a pinned Leaguepedia Cargo snapshot and direct Oracle's Elixir CSV or XLSX files.
Each adapter has an explicit pre-game field allowlist plus the outcome label, a versioned canonicalization contract, source checksums, retrieval metadata, and machine-readable quarantine reasons.

## Canonical match record

A historical match contains:

- Stable match ID.
- Timezone-aware match start timestamp.
- League and tournament.
- Region, tournament level, and official-circuit status when known.
- Patch.
- Blue and red team IDs or stable names.
- Five ordered blue players and five ordered red players.
- Five ordered blue champion picks and five ordered red champion picks.
- Up to five bans per side when available.
- Stable team and player IDs when the source provides them.
- First-pick side independently from map side when known.
- Series ID, game number, and score before the target game when known.
- Champions unavailable because of earlier games in a Fearless Draft series when known.
- Blue-side win label.

Champion picks are ordered by player role as top, jungle, middle, bottom, and support.
They are not stored in draft-action order.
The first vertical slice stores normalized arrays as JSON text in DuckDB.
Future ingestion adapters may add source-specific staging tables, but only validated canonical records enter feature generation.
Aliases are data, not hard-coded application mappings.

## Time boundaries

Historical feature generation replays matches in ascending timestamp order.
For every timestamp, it first computes features for all matches beginning at that timestamp and only then applies their outcomes to state.
This prevents arbitrary match-ID ordering from leaking between simultaneous matches.

The state contains pre-match team Elo, player Elo, regional meta-Elo, team home-region history, inactivity, bounded recent team form, role and global champion outcomes, player-champion outcomes, role matchups, player experience, league side rate, global side rate, and optional uncertainty-aware Gaussian latent components.
Each feature row records the greatest source timestamp used by its state.
That timestamp must be null for an empty history or strictly earlier than the target timestamp.

Elo is read before the current result is applied.
Team form excludes the current match.
Champion statistics exclude the current match and every later match, including later matches on the same patch.
Player-champion statistics follow the same rule.
Bans are preserved and validated even when a baseline does not obtain useful signal from them.
Map side and first-pick order are separate inputs because Riot's 2026 First Selection rules allow them to differ.
Fearless Draft context contains only champions used in earlier games of the same series.
The current target game's result or picks may never update that context before its feature row is captured.

Feature generation and state replay use exact timestamps rather than patch ordering.
Patch is context, not a time boundary.

A domestic match may use its current pregame event region for both teams.
An international match may use only each team's most recently observed domestic region from an earlier timestamp group.
Only explicitly official Primary or Premier cross-region results update regional ratings.
Unknown home regions stay unknown and force neutral regional ratings and evidence counts.
Regional ratings, evidence counts, home-region assignments, and their provenance timestamps are serialized in feature state.

An artifact state may be augmented for a currently running series only with completed games that began after the artifact cutoff and strictly before the target game.
That augmentation must use the same canonical validation and update logic as historical replay.
It may not consume statistics from the current game.

## Dataset splitting

Train, validation, and test partitions are contiguous chronological intervals.
Timestamp groups are indivisible so simultaneous matches cannot cross a split boundary.
The default fixture experiment uses 60 percent for training, 20 percent for validation, and 20 percent for test.
Release experiments may instead use fixed timezone-aware validation and test start timestamps.

The training interval has an internal chronological fit and calibration boundary.
Model preprocessing and estimators are fitted on the earlier fit portion only.
Probability calibration is fitted on the later calibration portion.
Candidate selection uses validation log loss.
The test interval is not used for fitting, calibration, or model selection.
When a release config enables pre-test refitting, the selected family is refitted on all pre-test data with a new trailing calibration interval before the test is evaluated once.

This design gives each standard release interval a single purpose:

- Fit interval: preprocessing and estimator parameters.
- Calibration interval: probability mapping only.
- Validation interval: model selection by log loss.
- Test interval: final chronological holdout reporting.

V3 model development adds expanding rolling-origin folds inside data ending in 2025.
Each fold has its own fit, trailing calibration, validation, and test interval.
The locked candidate is then measured once on an opened January through March 2026 diagnostic without evaluating the later interval.
The next promotion holdout begins after July 30, 2026.
V4 reuses those exact temporal boundaries for a matched partially pooled regional-strength experiment.

## Feature contract

The feature schema is auditable and combines point-in-time numeric state with optional categorical context.
It includes pre-match Elo values and difference, inactivity-adjusted team and role-player ratings, regional meta-Elo and cross-region evidence, team-plus-region pooled Elo, Elo implied probability, recent team form, historical team sample counts, league and global blue-side priors, role and global champion strength, player-champion strength and coverage, roster experience, roster continuity, matchup history, ban context, and parsed patch numbers.
It also represents first-pick order, game number, pre-game series score, and Fearless Draft availability when those fields are known.
The v6 extension adds an uncertainty-integrated latent probability, latent log-odds mean and standard deviation, additive side, region, team, player, roster, champion, patch-champion, and matchup contributions, roster uncertainty, and observed-component coverage.
Categorical CatBoost ablations may consume role-aligned team, player, champion, player-champion, league, region, tournament-level, patch, and ban identifiers.

Blue and red versions of equivalent numeric inputs pass through one preprocessing pipeline.
There are no independently fitted blue and red scalers.
Unknown entities use documented neutral priors and produce warnings.

The feature builder is the single implementation used for:

- Historical training rows.
- Holdout evaluation rows.
- Feature-state snapshots.
- Batch prediction.
- Live prediction.

## Models

The required controls are:

1. Historical blue-side base rate.
2. Elo-only probabilistic model.
3. Regularized logistic regression.
4. Histogram gradient-boosted trees.

V3 also compares feature-subset and role-expanded controls, recency weighting, native and explicitly calibrated CatBoost variants, raw categorical context, and fixed Elo-CatBoost ensembles.
V4 retains every v3 candidate on its exact pre-regional feature contract and adds regional Elo logistic, team-roster regional logistic, full regional logistic, regional histogram-tree, native regional CatBoost, and a fixed raw Elo-regional CatBoost ensemble.
V6 retains v4 on its exact 155-field contract and tests a diagonal-Gaussian dynamic paired-comparison filter, a mapped raw probability, one fixed augmented logistic model, and two fixed blends with v4.
All candidates emit finite bounded probabilities.
Calibration is learned only from the chronological calibration portion when both outcome classes are available.
Native variants optimize log loss directly and remain uncalibrated when that is the declared ablation.
Platt, monotone beta, and isotonic mappings are explicit matched candidates.
An output-calibrated ensemble fits both raw components on the estimator-fit interval and fits its final mapper only on the later combined calibration predictions.
Fixed ensembles record the calibration status and weight of each component.
No neural model is part of the current architecture.
A neural or Siamese candidate may be proposed only after repeated temporal backtests demonstrate a meaningful and stable log-loss improvement over these baselines.

## Evaluation

Log loss is the selection metric.
Every report also contains Brier score, expected calibration error, calibration bins, ROC-AUC when both classes exist, accuracy, and sample count.
The holdout report includes breakdowns by calendar month, patch, league, region, tournament level, and official status.
Small one-class groups report null ROC-AUC rather than inventing a value.

Results from synthetic fixtures validate the workflow, not real-world predictive quality.
Production claims require public-data backtests across multiple seasons and leagues.
Repeated rolling-origin folds are the primary development evaluation because a single holdout can be unusually favorable or unfavorable.
The final untouched chronological holdout remains the only source for a release claim.
Uncertainty intervals are clustered by series so games from one best-of series are not treated as independent observations.
Release reports also contain a paired series-clustered interval for candidate-minus-Elo log loss.
Promotion applies fixed aggregate, calibration, sample-size, probability-contract, and major-league gates.
A failed gate selects the configured simpler fallback rather than allowing aggregate accuracy to override the failure.

## Nested optimization boundary

The optimizer uses a dedicated DuckDB loader that applies an exclusive supervised cutoff inside SQL.
Rows at or after that cutoff never enter optimizer memory.
Feature-replay trials use the same exclusive SQL boundary when loading canonical matches.

Four outer expanding-window folds estimate the complete search policy.
Three inner expanding-window folds rank candidates inside each outer history.
Estimator preprocessing, identity vocabularies, fit coefficients, and output calibration use only their assigned earlier partitions.
An outer outcome is evaluated only after that outer fold's inner winner is immutable.

Trial specifications are generated deterministically from the frozen seed.
The schedule balances outer folds, model families, and cached or replay feature modes.
A standard-library SQLite registry stores immutable specifications and hashes, status, timings, sanitized failures, and inner metrics.
Only the orchestrator writes SQLite, while worker processes return results through process futures.
An exclusive file lock prevents two orchestrators from mutating one study.

The registry refuses a resume when the bounded dataset, resolved configuration, schedule, cutoff, or optimizer code fingerprint changes.
Interrupted running trials return to pending on a locked resume.
Completed trials are never submitted again.
The final specification is written to an immutable lock file before any outer evaluation begins.

## Hierarchical state-space boundary

The optional v6 filter represents global blue side, league blue side, region, team, player, exact team roster, role champion, patch-role-champion, and antisymmetric role-matchup terms as centered Gaussian components.
Each component stores a mean, variance, observation count, and last-observed timestamp.
New or unknown entities use zero mean and the configured component prior variance.

At a target timestamp, means decay toward zero and posterior variances return toward their prior uncertainty.
The raw probability integrates the diagonal linear-predictor variance through a fixed logistic approximation.
High uncertainty therefore pulls a forecast toward 50 percent.

Every match at one timestamp reads one immutable projected state.
The complete timestamp group accumulates deterministic per-component gradients and diagonal curvature before posterior means and variances change.
Observation IDs are sorted before accumulation, so input order within one group cannot change state.

The filter updates only from explicitly official `Primary` or `Premier` games.
Domestic event metadata may establish a team's home region before a domestic match.
An international region term may use only a home region established at an earlier timestamp.

V6 uses four outer and three inner expanding-window folds under an exclusive pre-2026 storage cutoff.
Only five preregistered candidates are eligible for selection.
The fixed v4 control cannot consume the appended state-space fields.
If no challenger satisfies inner log-loss, ECE, fold, and major-league rules, v4 is the automatic fallback.

The outer policy must satisfy a material log-loss threshold, a paired series-clustered interval, calibration, sample size, fold robustness, and major-league robustness before it can replace v4.
The completed v6 study failed the paired-interval and LCS gates, so the state-space artifact remains development-only.

## Artifact format

An artifact is a versioned directory that contains:

- `manifest.json` with artifact schema version, artifact purpose, model version, model kind, feature schema version, feature names, creation time, data cutoff, split ranges, training config hash, dependency versions, selection rule, calibration provenance, and an optional locked model specification.
- `model.joblib` with the fitted estimator and optional probability calibrator.
- `feature-state.json` with the point-in-time state at the training cutoff, known entity sets, and optional latent means, variances, counts, and timestamps.
- `metrics.json` with candidate validation and holdout results.
- `training-config.json` with the resolved experiment configuration.
- `checksums.json` with SHA-256 hashes for artifact payloads.

The loader verifies the schema version, required files, hashes, and feature contract before deserializing the local model.
Serialized model files are trusted local build outputs and must never be loaded from an untrusted source.
Models, artifacts, databases, reports, and registry state are ignored by Git.

Evaluation artifacts preserve their untouched chronological test interval and its feature-state cutoff.
They are the only artifacts accepted by the historical evaluation command.
Development artifacts refit a locked rolling-origin selection for shadow use after an opened diagnostic.
They carry `artifact_purpose: development`, never claim prospective promotion, and add an explicit warning to every prediction.
Production artifacts refit the gate-recommended family after evaluation and replay feature state through all available games.
They carry `artifact_purpose: production` and cannot be presented as untouched evaluation artifacts.

## Prediction contract

The JSON request contains:

- `request_schema_version`
- `match_timestamp`
- `league`
- `tournament`
- optional `region`
- optional `tournament_level`
- optional `is_official`
- `patch`
- optional `series_id`
- `game_number`
- `blue_series_wins_before`
- `red_series_wins_before`
- `blue_team`
- `red_team`
- optional stable team IDs
- `blue_players`
- `red_players`
- optional stable player IDs
- `blue_picks`
- `red_picks`
- optional `blue_bans`
- optional `red_bans`
- optional `first_pick_side`
- optional `fearless_bans`

The prediction timestamp must be strictly later than the artifact data cutoff.
This prevents a current-state snapshot from being used for an earlier historical match.
Historical backtests use replayed historical feature rows instead.
Unknown first-pick or series fields are permitted for older data and produce missing-context warnings rather than guessed values.
Every request model forbids extra fields, so an accidental `gold`, `kills`, or similar in-game field fails validation.

The JSON response contains:

- `blue_win_probability`
- `red_win_probability`
- `model_version`
- `data_cutoff_timestamp`
- `warnings`

Warnings identify unknown teams, players, champions, home regions, leagues, patches, and stale artifacts.
The two probabilities must be finite, fall in `[0, 1]`, and sum to one within floating-point tolerance.

## Screenshot contract

An overlay profile is a versioned JSON document containing normalized crop rectangles for two team identifiers, ten role-aligned champion portraits, and optional ban portraits.
Profiles are specific to a tournament overlay and layout version.
A template catalog maps trusted local reference images to canonical team or champion names.

Each parsed field contains:

- Best candidate value.
- Confidence.
- Similarity to the best template.
- Runner-up value and similarity.
- Disposition of accepted, review, or unreadable.

The parser returns no win probability when any required field is below the configured acceptance threshold.
A user correction or explicit confirmation creates the final `DraftRequest`.
The prediction process stores the screenshot checksum, parser version, profile version, catalog checksum, and confirmed structured request for auditability.

Screenshot parser quality and model quality are evaluated separately.
The parser benchmark is split by match, recording, and overlay version so adjacent frames from one broadcast cannot appear in both development and test data.
The initial supported-overlay targets are at least 99.5 percent precision for automatically accepted individual slots and at least 95 percent exact-match accuracy for all ten champion slots.
Low-confidence abstention is preferable to a confidently wrong draft.

The screenshot benchmark must include crops with animation, compression, blur, scaling, color shifts, substitutions, mirrored layouts, and unrelated in-game pixels outside the configured draft regions.
Changing pixels outside configured regions must not change parsed draft fields.

Every real benchmark frame records a source URL, license, redistribution-rights status, image checksum, resolution, video timestamp, recording group, match group, profile, catalog, partition, and verified structured labels.
Raw frames remain outside Git unless redistribution rights are explicit.
Manifest validation prevents a recording or match group from crossing development and holdout partitions.
Unsupported layouts are labeled and must not be accepted automatically.

## Reproducibility and operations

Runtime and development dependencies are locked with `uv.lock`.
Experiment settings live in versioned configuration files.
The local artifact directory acts as the first reproducible experiment registry.
An MLflow adapter can be added later without changing model or feature contracts.

CI runs formatting, linting, type checking, unit tests, integration tests, secret guards, and the full fixture-backed workflow.
No CI step requires a Riot credential or private dataset.

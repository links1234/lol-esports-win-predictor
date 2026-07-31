# V5 leakage-safe optimization protocol

## Status

This protocol is frozen before the v5 optimizer is allowed to read any supervised outcome.
It defines a development experiment and cannot promote a production model.
Every outcome currently available in 2026 has already been opened by earlier development work, so no 2026 result may influence a trial, a stopping decision, a model choice, or a feature choice.

## Objective

The objective is to improve calibrated pregame win probabilities for professional League of Legends drafts.
Primary selection uses log loss because the product must provide useful probabilities rather than only correct binary classifications.
Brier score, expected calibration error, ROC-AUC, accuracy, sample count, time, patch, league, region, and tournament breakdowns remain diagnostics.

The optimizer compares auditable model and feature hypotheses under nested chronological validation.
It does not search neural architectures.
A neural or Siamese model remains out of scope until a simpler candidate demonstrates stable value and supplies a credible benchmark.

## Frozen data boundary

The exclusive supervised development cutoff is `2026-01-01T00:00:00Z`.
The optimizer must query DuckDB with this cutoff and must never load a 2026 label into its process.
The source feature table may contain later rows, but those rows are inaccessible through the optimizer's bounded storage API.
Feature-replay trials must likewise load only source matches strictly earlier than the cutoff.

The modeling population is limited to explicitly official `Primary` or `Premier` matches.
Rows remain grouped by exact match timestamp so simultaneous games cannot observe one another.
Series remain intact for uncertainty estimates.

The January through March 2026 diagnostic is already opened and is not optimization evidence.
The April through July 2026 interval is also unavailable to this experiment.
The final v5 result remains a shadow recommendation until genuinely new prospective data satisfies the existing promotion gates.

## Nested chronological design

Four expanding-window outer folds estimate the complete search policy.
Each outer test interval is touched only after all candidates for that outer fold have been ranked using three inner expanding-window folds.
The outer history contains only rows strictly earlier than that outer test interval.

Each inner fold has an estimator-fit interval, a later probability-calibration interval when required, and a still later scoring interval.
Preprocessing, imputation, scaling, identity vocabulary construction, model fitting, and probability calibration are fitted only inside the relevant inner history.
No random train-test split is permitted.

The optimizer chooses one candidate independently inside each outer fold.
That candidate is refitted on the complete outer history using a trailing calibration partition when its output mapper requires one.
It is then scored once on the outer test interval.
Pooled outer predictions estimate the performance of the selection procedure rather than the performance of a candidate chosen after seeing every outer result.

The locked v5 finalist is the candidate chosen from the fourth outer fold's inner results.
The fourth outer outcome may characterize that locked candidate but may not change it.

## Trial budget and ordering

The run stops after 300 completed trials or 12 elapsed hours, whichever comes first.
Four worker processes may run concurrently.
Each model fit is limited to one internal thread so the four-worker limit remains meaningful.

Each outer fold receives 75 deterministic trial specifications.
Each outer allocation contains 50 cached-feature trials and 25 full point-in-time feature-replay trials.
Each outer allocation contains 15 trials for each of the five model families.
Within each family, ten trials use cached features and five trials replay feature state.

Trial submission is round-robin across outer folds, model families, and replay modes.
The trial schedule is generated from the frozen seed `20260730` before scoring.
A timeout therefore leaves a balanced prefix rather than a family-biased search.

The native study registry uses SQLite from the Python standard library because Optuna and MLflow are not installed in the locked environment.
The registry records deterministic trial specifications, fingerprints, status, timing, metrics, warnings, and errors.
It supports safe resumption and refuses to combine a study with a different dataset, configuration, cutoff, schedule, or code contract.
This implementation leaves a narrow registry interface so a later Optuna adapter can be added without changing the temporal experiment.

## Feature search

The search selects coherent feature groups rather than treating the 155 numeric features as independent switches.
Core side, team-strength, and data-coverage fields are always present.
Optional groups cover roster and form, champion meta, player-champion history, draft interactions, regional strength, and categorical pregame context.

Full replay trials may choose only from the following frozen point-in-time state parameters:

| Parameter | Values |
| --- | --- |
| Team Elo K | 12, 18, 24, 30, 36 |
| Player Elo K | 6, 9, 12, 16, 20 |
| Regional Elo K | 6, 9, 12, 18, 24 |
| Team-form window | 5, 8, 10, 15, 20 |
| General rate prior | 6, 9, 12, 18, 24 |
| Player-champion prior | 4, 6, 8, 12, 16 |
| Synergy prior | 12, 18, 24, 32 |
| Matchup prior | 10, 15, 20, 30 |
| Rating inactivity half-life in days | 120, 240, 365, 540, 730 |

Every replay starts from empty state and processes only matches earlier than the development cutoff.
The historical and live feature builders remain the same functions.
Target labels are applied only after every match at the target timestamp has been featurized.

## Model families

The frozen families are regularized logistic regression, histogram gradient-boosted trees, CatBoost, an Elo-CatBoost probability blend, and dynamic Bradley-Terry.

Logistic regression searches L2 strength and coherent feature groups.
Histogram trees search learning rate, iteration count, leaf count, L2 regularization, and coherent feature groups.
CatBoost searches iterations, depth, learning rate, L2 leaf regularization, random strength, column sampling, feature groups, and optional pregame categorical context.
CatBoost retains ordered input semantics with `has_time=True`, plain boosting, deterministic seeds, and one thread per fit.
The blend searches a fixed convex Elo weight for each trial and never estimates its weight on an outer outcome.

Dynamic Bradley-Terry uses signed blue and red team, player, champion, league-side, and region-side terms.
Its identity vocabulary, numeric scaling, and regularized logistic coefficients are fitted from the trial fit interval only.
Older fit rows receive deterministic exponential recency weights whose reference time is the end of that fit interval.
Unknown future identities map to neutral zero contribution.
The family searches L2 strength, recency half-life, player weight, champion weight, and coherent numeric feature groups.

The model ranges are:

| Parameter | Range |
| --- | --- |
| Logistic C | log-uniform from 0.01 to 5 |
| Histogram learning rate | log-uniform from 0.015 to 0.12 |
| Histogram iterations | integer from 100 to 400 |
| Histogram leaves | integer from 7 to 31 |
| Histogram L2 | log-uniform from 0.1 to 20 |
| CatBoost iterations | integer from 150 to 700 |
| CatBoost depth | integer from 4 to 8 |
| CatBoost learning rate | log-uniform from 0.02 to 0.1 |
| CatBoost L2 | log-uniform from 3 to 30 |
| CatBoost random strength | uniform from 0 to 1.5 |
| CatBoost column sampling | uniform from 0.65 to 1 |
| Elo blend weight | uniform from 0.25 to 0.75 |
| Bradley-Terry C | log-uniform from 0.005 to 2 |
| Bradley-Terry recency half-life | log-uniform from 120 to 900 days |
| Bradley-Terry player weight | uniform from 0.25 to 1.5 |
| Bradley-Terry champion weight | uniform from 0.1 to 1.25 |

Native, Platt, beta, and isotonic output mappings may be tested where the family supports them.
Every learned mapper uses only the trailing calibration interval.
The same mapper is applied unchanged to later scoring rows.

## Controls and selection

Every inner and outer evaluation includes Elo-only, team-roster logistic, and the current v4 regional Elo-CatBoost shadow model as fixed controls.
Controls use the same temporal partitions as searched candidates.

Within one outer fold, trials are ordered by pooled inner log loss.
A trial is eligible only if all of these checks pass:

- Its pooled inner expected calibration error is at most 0.04.
- Its pooled inner log loss beats Elo.
- Its pooled inner log loss beats the best fixed simple control.
- It does not regress against Elo in any inner fold.
- Every configured major-league group with at least 100 examples stays within 0.01 log loss of Elo.

The eligible trial with the lowest pooled inner log loss wins.
If no searched trial is eligible, the current v4 shadow configuration is the fallback for that outer fold.
Outer scores never participate in hyperparameter choice.

Pooled outer results include a series-clustered 95 percent interval and paired differences against Elo and the fixed v4 control.
No success claim may rely on accuracy alone.
A small log-loss gain with an interval crossing zero is reported as uncertain.

## Reproducibility and artifacts

The study directory contains the SQLite registry, the resolved optimization configuration, fingerprints, an incremental status report, the final nested report, and a locked finalist specification.
Generated study state stays under ignored `var/` paths.
No model, database, raw data, registry, report, environment file, credential, or API key enters Git.

The registry fingerprint covers the bounded pre-2026 dataset, the base experiment configuration, the optimization configuration, the feature-group contract, and the trial schedule.
Each trial specification has a stable canonical JSON hash.
Repeating specification generation with the same seed must produce byte-identical JSON.

## Acceptance criteria

- Storage tests prove that rows and matches at or after the exclusive cutoff are not returned.
- Split tests prove strict fit, calibration, inner-score, and outer-score ordering with indivisible timestamps.
- Target or future-outcome mutations cannot alter an earlier feature row or trial specification.
- Preprocessing and identity vocabularies are fit only from the trial fit interval.
- Full replay and cached default features are identical for the frozen default settings.
- Trial schedules are deterministic and balanced across outer folds, families, and replay modes.
- Interrupted trials can be resumed without duplicating completed work.
- Registry fingerprint mismatches fail closed.
- Failed trials record sanitized errors and do not corrupt the study.
- Four workers cannot cause nondeterministic trial specifications or SQLite write races.
- Synthetic end-to-end optimization creates a registry, resumes it, selects finalists, and writes a valid report.
- Ruff, formatting, strict mypy, build, repository hygiene, credential scanning, and the full test suite pass.
- The real run either completes 300 trials or records that the 12-hour deadline stopped a balanced prefix.
- The final report contains metrics, cutoff, uncertainty, breakdowns, controls, winner rationale, failure counts, and all commands needed to reproduce the run.

## Interpretation boundary

This search can find the best tested configuration under the frozen historical policy.
It cannot prove that the model is universally best, guarantee a live-game result, or turn an observational historical dataset into certainty.
Its output remains a preliminary pregame probability based on known teams, rosters, champions, sides, tournament context, and bans.
Screenshot parsing quality remains a separate measured system and must not be conflated with probability-model accuracy.

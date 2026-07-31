# V5 leakage-safe optimization results

## Status and decision

The v5 nested optimization study is complete.
All 300 scheduled trials completed with zero failures in 2,943.71 active seconds, or approximately 49 minutes and 4 seconds.
The run stopped on the full trial budget rather than the 12-hour limit.

The v5 search did not produce evidence that justifies replacing the current v4 regional Elo-CatBoost shadow model.
The nested v5 selection policy scored 0.610721 log loss, while the fixed v4 control scored 0.609566 on the same pooled outer intervals.
The v5-minus-v4 point estimate is therefore a 0.001156 regression.
Its series-clustered 95 percent interval runs from -0.002386 to 0.004902, so the data is compatible with either a small improvement or a larger small regression.

The current v4 shadow model remains the recommended probability model.
The locked v5 challenger is retained only as a reproducible development artifact.
Neither model has passed a prospective promotion gate.

## Frozen boundary and population

The study followed [`docs/v5-optimization-protocol.md`](v5-optimization-protocol.md).
The exclusive supervised development cutoff was `2026-01-01T00:00:00Z`.
The optimizer queried DuckDB with that cutoff and never loaded a 2026 label into an optimizer process.

The bounded source contained 84,862 matches before the cutoff.
The official `Primary` or `Premier` modeling population contained 20,380 games.
Its final eligible timestamp was November 9, 2025.
The four outer score intervals contained 8,135 games from 3,582 series and ran from March 21, 2023 through November 9, 2025.

The January through March 2026 diagnostic remained excluded from tuning.
The April through July 2026 interval also remained excluded from tuning and nested evaluation.
No 2026 result influenced a trial, feature choice, hyperparameter choice, stopping decision, or finalist choice.

## Search execution

The search used four worker processes.
Each model fit used one internal thread.
The deterministic schedule assigned 75 trials to each outer fold.
Each outer fold received ten cached and five replay trials for each of five model families.

The five families were regularized logistic regression, histogram gradient-boosted trees, CatBoost, an Elo-CatBoost blend, and dynamic Bradley-Terry.
One hundred trials replayed the full point-in-time feature state with searched feature parameters.
Two hundred trials used the frozen cached point-in-time feature table while searching model parameters and coherent feature groups.

The SQLite registry checkpointed every result.
A complete resume performed no duplicate trial.
Every trial attempt count remained exactly one.
The locked-finalist file remained byte-identical across the resume.

## Nested outer-policy comparison

Log loss is the primary metric.

| Candidate or policy | Log loss | Brier | ECE | ROC-AUC | Accuracy | Games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current v4 regional Elo-CatBoost control | **0.609566** | **0.211037** | 0.018571 | **0.725704** | **66.27%** | 8,135 |
| V5 nested selection policy | 0.610721 | 0.211409 | 0.013181 | 0.723875 | 66.18% | 8,135 |
| Team-roster logistic control | 0.611943 | 0.212361 | **0.009626** | 0.719953 | 65.42% | 8,135 |
| Elo-only control | 0.620256 | 0.215778 | 0.019824 | 0.711530 | 66.01% | 8,135 |

The v5 policy improved Elo by 0.009535 log loss.
Its paired interval against Elo is -0.014594 to -0.004710, which stays below zero.
The v5 policy improved team-roster logistic by 0.001221, but its interval of -0.005115 to 0.002714 crosses zero.
The v5 policy regressed against v4 by 0.001156, and its interval of -0.002386 to 0.004902 crosses zero.

The selected policy's own series-clustered 95 percent log-loss interval is 0.601243 to 0.619925.
Its ECE is lower than v4's ECE, but calibration improvement does not override worse primary log loss.
Its ROC-AUC and accuracy are also slightly below v4.

## Outer selections

Each outer winner was selected from inner results before its outer outcome was touched.

| Outer fold | Inner-selected family | Feature mode | Mapping | Inner log loss | Outer log loss | Outer games |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Logistic regression | Cached | Native | 0.610784 | 0.611052 | 2,048 |
| 2 | Logistic regression | Cached | Native | 0.612459 | 0.596634 | 2,030 |
| 3 | Logistic regression | Replay | Native | 0.604981 | 0.608076 | 2,034 |
| 4 | Elo-CatBoost blend | Replay | Beta | 0.598537 | 0.627183 | 2,023 |

The first three outer selections favored logistic regression.
The fourth favored the replayed Elo-CatBoost blend by only 0.000503 inner log loss over dynamic Bradley-Terry and by 0.000702 over logistic regression.
This changing winner pattern is evidence of temporal model-selection instability.
It argues against treating one narrow inner-fold win as a universal architecture result.

The fixed v4 control scored 0.624296 on the fourth outer interval.
The locked v5 challenger scored 0.627183, a 0.002887 regression.
The paired fourth-outer candidate-minus-v4 interval is -0.000760 to 0.006393 across 771 series.
The locked challenger improved fourth-outer ECE from 0.028083 to 0.024589, but it had worse log loss, Brier score, ROC-AUC, and accuracy.

## Family findings

Logistic regression was the most consistently eligible searched family.
It supplied 9, 6, 5, and 1 eligible trials across the four outer searches.
Dynamic Bradley-Terry supplied 5, 7, 1, and 1 eligible trials and was often close to the best logistic model.
Histogram trees supplied 2, 1, 0, and 0 eligible trials.
CatBoost supplied 1, 1, 1, and 0 eligible trials.
The Elo-CatBoost blend supplied 0, 4, 0, and 1 eligible trials.

These results do not show a stable advantage for higher-capacity trees.
They also do not establish that the new dynamic Bradley-Terry family is better than the simpler models.
The strongest practical finding is that careful temporal state, team strength, and regularization matter more consistently than model complexity.

The selected policy improved v4 in several smaller domestic groups but regressed in important international and recently reorganized competitions.
Its pooled region regression versus v4 was 0.015452 for International and 0.014207 for Americas.
It improved Asia Pacific by 0.005110, Brazil by 0.008351, and Vietnam by 0.008317.
The 2025 First Stand group contained only 35 games and regressed by 0.216505, so it is a serious warning but not a stable estimate.

## Locked v5 challenger

The final specification was selected from the fourth outer fold's inner results only.
Its lock fingerprint is `6df8b926e54c20b099e31e3cc0bf81111765f44d92910df883ac5b0ab60442ab`.

The challenger uses an Elo-CatBoost blend with beta calibration.
Its Elo weight is 0.284886 and its CatBoost weight is 0.715114.
It uses core, team-strength, roster-form, draft-interaction, and regional feature groups.
It excludes the champion-meta, player-champion, and categorical-identity groups.

Its feature-state parameters are:

- Team Elo K of 18.
- Player Elo K of 20.
- Regional Elo K of 12.
- Team-form window of 15 games.
- General rate prior strength of 6.
- Player-champion prior strength of 12.
- Synergy prior strength of 24.
- Matchup prior strength of 15.
- Rating inactivity half-life of 540 days.

Its CatBoost parameters are:

- 593 iterations.
- Depth 6.
- Learning rate 0.0246136.
- L2 leaf regularization 19.9075.
- Random strength 1.16089.
- Column sampling rate 0.928126.

## Development artifact

After the specification was locked, it was refit as a development artifact on all already-available data.
This post-lock refit may use already-opened 2026 outcomes to estimate coefficients and current feature state, but those outcomes did not revise the specification.
The artifact therefore supports current shadow predictions but does not create new evaluation evidence.

The artifact is:

```text
var/v5-optimization/artifacts/leaguepedia-2020-2026-v4-regional-v5-locked-ac748c91-development-v5_elo_catboost_blend_ac748c91ac35-20260731T052352Z-39402d84
```

It contains 19,953 estimator-fit games and 2,209 trailing beta-calibration games.
Its feature-state cutoff is July 30, 2026 at 19:52 UTC.
Its model version is `v5-locked-ac748c91ac35+cfg.39402d84.data.f4e69cdb`.
Its manifest embeds the complete locked trial specification and development-only status.
Every prediction includes the development-model warning.

A separate process loaded the artifact and predicted the saved hypothetical LCK draft.
It returned 67.3312 percent blue and 32.6688 percent red.
The existing v4 artifact returned 67.4466 percent blue on that same hypothetical request.
Neither value is a fact, guarantee, or evaluation result.

## Preserved and deliberately discarded

The optimizer preserves DuckDB, explicit point-in-time replay, simultaneous-timestamp isolation, chronological evaluation, configuration-driven experiments, coherent feature groups, shared training and prediction features, model calibration, series-clustered uncertainty, fixed controls, and checksummed artifacts.
It adds deterministic four-worker scheduling, a resumable single-writer SQLite registry, bounded SQL reads, dynamic Bradley-Terry, full feature-state search, immutable trial hashes, and an inner-only finalist lock.

It deliberately avoids random train-test splits, full-dataset or same-patch aggregates, final historical Elo reuse, independent blue and red preprocessing pipelines, hard-coded team mappings, unsafe evaluation, accuracy-only selection, unbounded identity fitting, post-2025 tuning, and neural-model escalation.
It also refuses to promote the searched challenger merely because some metrics or subgroups improved.

## Verification

The complete suite passes 113 tests.
Statement coverage is 90.23 percent against the 85 percent CI floor.
The four-worker synthetic optimizer completes 30 trials with zero failures, resumes without duplicate attempts, refits a locked artifact, loads it in a separate prediction process, and produces a valid warned probability.
Ruff formatting and linting, strict mypy, lock validation, source and wheel builds, repository hygiene, credential-pattern scanning, bounded-storage tests, model-vocabulary tests, and artifact checksum loading pass.

The locked-finalist SHA-256 is `8d82f7fc63df46588de4288f1078ac8120967fbdefe6190be8263f999b95fa61`.
The nested-report SHA-256 is `6e7ec7749e375e234bd530258bd839e297dc41be337b7893e6a3c2c842f33166`.
The development-refit report SHA-256 is `b0dd6f817aaa8a1d668d7d9b1831de893ea611f5c38f51698f2172a33df0eb41`.
The sample prediction SHA-256 is `7384aec97dffe938eaaeb2f3067e829cf7e643b8b0728ccff38153dd1130a25e`.

## Exact commands

Install the locked environment:

```bash
uv sync --locked --all-groups
```

Run or resume the complete real study:

```bash
uv run lolpredictor optimize \
  --database var/v4-regional/matches.duckdb \
  --config configs/v5-optimization.yaml \
  --study var/v5-optimization
```

Read progress without changing the study:

```bash
uv run lolpredictor optimize-status \
  --study var/v5-optimization
```

Refit the locked challenger as a development artifact:

```bash
uv run lolpredictor optimize-refit \
  --database var/v4-regional/matches.duckdb \
  --config configs/v5-optimization.yaml \
  --locked-finalist var/v5-optimization/locked-finalist.json \
  --nested-report var/v5-optimization/nested-report.json \
  --registry var/v5-optimization/artifacts \
  --output var/v5-optimization/refit-report.json
```

Run a structured prediction with the development challenger:

```bash
uv run lolpredictor predict \
  --artifact var/v5-optimization/artifacts/leaguepedia-2020-2026-v4-regional-v5-locked-ac748c91-development-v5_elo_catboost_blend_ac748c91ac35-20260731T052352Z-39402d84 \
  --input examples/lck_hypothetical_draft.json \
  --output var/v5-optimization/sample-shadow-prediction.json
```

Run the synthetic optimizer workflow:

```bash
uv run lolpredictor fixture \
  --database var/v5-fixture/fixture.duckdb

uv run lolpredictor features \
  --database var/v5-fixture/fixture.duckdb \
  --config configs/v5-optimization-fixture-base.yaml

uv run lolpredictor optimize \
  --database var/v5-fixture/fixture.duckdb \
  --config configs/v5-optimization-fixture.yaml \
  --study var/v5-fixture/study
```

Run all quality checks:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv build
uv run pytest \
  --cov=src/lolpredictor \
  --cov-report=term-missing \
  --cov-fail-under=85
```

## Remaining risks and next milestone

The existing v4 model remains a shadow model because no untouched post-July 2026 promotion interval exists.
The v5 result shows that more hyperparameter trials alone are unlikely to create a reliable step change.
Winner families changed across time, the locked inner advantage did not survive its outer period, and several new competition groups were unstable.

The next highest-value modeling milestone is a preregistered hierarchical state-space strength model rather than a larger generic tree search.
It should model latent team strength, player contributions, roster transitions, region strength, seasonal drift, and uncertainty jointly.
Champion and matchup effects should use role-aware patch hierarchies with partial pooling rather than sparse independent rates.
The model should retain the same timestamp-batched outcome updates, nested chronological policy evaluation, and fixed v4 controls.

The highest-value evidence milestone remains a genuinely new prospective holdout.
Promotion still requires at least 750 new games from at least 300 series, acceptable ECE, no major-league gate failure, and a paired log-loss improvement that does not rely on accuracy.

The screenshot milestone remains separate.
A rights-cleared, recording-grouped corpus must demonstrate team and champion extraction precision, exact-draft accuracy, and safe abstention before screenshot input is treated as unattended.

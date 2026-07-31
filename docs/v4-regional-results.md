# V4 partially pooled regional-strength results

## Status and decision

The v4 regional experiment is complete for development and shadow prediction.
The selected model is a fixed 50 percent raw Elo and 50 percent native regional CatBoost ensemble.
It is a development model and not a promoted production model.

The experiment followed `docs/v4-regional-protocol.md`.
The candidate family and regional K factor were fixed before the new outcome comparison.
Selection used only rolling test intervals ending on November 9, 2025.
Only the locked candidate was then evaluated on the already-opened January through March 2026 diagnostic.
No April through July 2026 outcome was evaluated by this workflow.

## Source and metadata audit

The pinned Leaguepedia source is `data/raw/leaguepedia-2020-2026-through-20260730T234000Z-v2.json`.
Its SHA-256 is `c0f94d5982d7f8ddb82ea7c3de6203a5d19232ba3a13157f9ce927dbcedf9db0`.
It contains 95,588 source games, of which 93,041 were accepted and 2,547 were quarantined.
The accepted feature history ends on July 30, 2026 at 19:52 UTC.

The official Primary or Premier modeling population contains 22,162 games.
It contains 22 league labels and 15 region labels with no missing target-population region or tournament-level values.
Every eligible game is labeled `Primary`.
There is no `Premier` example and no tier variation from which to estimate a tournament-tier effect.
The experiment therefore did not expand into Secondary events or manufacture a constant tier feature.

The target population contains 1,245 international-event games.
The point-in-time home-region replay identifies 1,104 cross-region games.
Of those rows, 1,101 have a nonzero regional rating difference from earlier eligible cross-region results.
All regional provenance timestamps are strictly earlier than their target matches.

## Implemented point-in-time contract

A domestic request uses its known non-international event region for both teams.
An international request uses each team's most recently observed domestic region from an earlier timestamp group.
No team name or team ID is hard-coded to a region.
Unknown home regions produce neutral ratings, neutral evidence counts, and an explicit prediction warning.

Every regional rating starts at 1500 and uses a fixed K factor of 12.
Only explicitly official Primary or Premier matches with two known and different home regions update the ratings.
All matches at one timestamp read the same pre-update state.
Their rating deltas are accumulated and applied only after the complete timestamp group has been featurized.
Unofficial, Secondary, Showmatch, and missing-level matches cannot move regional ratings.

The v5 contract adds regional ratings, rating difference, implied probability, cross-region evidence counts, coverage indicators, a cross-region indicator, and team-plus-region pooled Elo.
The same builder generates historical rows and live prediction rows.
Feature state, home-region state, rating evidence, and regional provenance survive artifact serialization.
Artifacts built with the 143-feature pre-regional contract remain loadable.

## Rolling comparison

The three rolling test intervals contain 6,087 games from 2,638 series.
They run from September 2, 2023 through November 9, 2025.
Log loss is the selection metric.

| Candidate | Log loss | Brier | ECE | ROC-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native regional CatBoost | **0.609354** | **0.210967** | 0.015544 | 0.725485 | 66.31% |
| 50% raw Elo + 50% native regional CatBoost | 0.609672 | 0.211143 | 0.023263 | **0.725861** | 66.34% |
| 50% raw Elo + 50% native numeric CatBoost, v3 control | 0.610845 | 0.211731 | 0.022764 | 0.723759 | 66.19% |
| Native numeric CatBoost, v3 control | 0.610922 | 0.211728 | 0.016061 | 0.723241 | 66.32% |
| Regional histogram trees | 0.612517 | 0.212334 | 0.015181 | 0.721573 | **66.60%** |
| Team-roster logistic | 0.612719 | 0.212665 | **0.009380** | 0.719184 | 65.65% |
| Regional Elo logistic | 0.619463 | 0.214571 | 0.020326 | 0.716691 | 66.01% |
| Elo only | 0.619977 | 0.215634 | 0.021001 | 0.712263 | 65.85% |
| Blue-side base rate | 0.691354 | 0.249104 | 0.005281 | 0.498837 | 53.05% |

The selected regional ensemble improves rolling log loss by 0.001173 over the matched v3 ensemble.
Its 95 percent series-clustered log-loss interval is 0.600887 to 0.619141.
Its paired candidate-minus-Elo interval is -0.013270 to -0.007394.
It improves Elo by 0.015155, 0.010738, and 0.005003 log loss in the three folds.

The selected model scores 0.603360, 0.601933, and 0.623786 in the three folds.
The matched v3 ensemble scores 0.603891, 0.602016, and 0.626701.
The regional ensemble therefore improves the matched v3 control in every fold, although two differences are small.

The clearest ablation effect is on international events.
The selected regional ensemble improves international log loss from 0.667656 to 0.645522 across 496 rolling games.
It slightly regresses several domestic-region breakdowns, including China, EMEA, and Asia Pacific.
That pattern supports the intended cross-region use case but does not establish a universal domestic improvement.

## Robustness selection

Native regional CatBoost is the raw aggregate metric winner.
It is not policy eligible because its LCS breakdown regresses 0.016482 versus Elo across 192 games, above the frozen 0.01 limit.

The fixed regional ensemble is eligible.
It beats Elo and the best simple control on aggregate log loss, beats Elo in every fold, has ECE below 0.04, has a paired interval below zero versus Elo, and stays within every fixed major-league regression limit.
Its LCS regression is 0.008707 versus Elo, which is still close to the 0.01 boundary.

The selected-minus-team-roster-logistic interval is -0.006516 to 0.000454.
Because that interval crosses zero, the evidence does not establish superiority over the simpler logistic control at the 95 percent level.

## Opened 2026 diagnostic

The locked model was refitted on all 20,380 eligible games through November 9, 2025.
It was evaluated on 744 games from January 14 through March 30, 2026.

The selected model scores 0.667748 log loss, 0.237538 Brier score, 0.063098 ECE, 0.637159 ROC-AUC, and 58.74 percent accuracy.
Elo scores 0.695109 log loss, 0.248127 Brier score, 0.072408 ECE, 0.612155 ROC-AUC, and 57.80 percent accuracy.
The paired selected-minus-Elo interval is -0.037179 to -0.016794 across 316 series.

The selected model improves the matched v3 ensemble's diagnostic log loss by 0.000509.
Its ECE is worse than the v3 ensemble's 0.054683 and remains above the fixed 0.04 target.
The diagnostic therefore confirms useful ranking signal but unresolved probability drift.
It is not promotion evidence and was not used for tuning.

## Shadow artifact

The warned development artifact is:

```text
var/v4-regional/artifacts/leaguepedia-2020-2026-v4-regional-development-elo_catboost_regional_raw_blend_50-20260731T032543Z-3dc6f839
```

Its model version is `v4-regional-20260730+cfg.3dc6f839.data.f4e69cdb`.
Its feature-state cutoff is July 30, 2026 at 19:52 UTC.
Both native components use all 22,162 eligible games and no calibration rows.
The manifest records equal weights, native output mapping, 155 model features, and `artifact_purpose: development`.
Every prediction includes a warning that the model has not passed prospective promotion.

A separate process loaded the artifact and evaluated `examples/lck_hypothetical_draft.json`.
It returned 67.4466 percent blue and 32.5534 percent red.
That value is a model estimate for a hypothetical saved draft, not a known fact or guaranteed win rate.

## Preserved and discarded

The experiment preserves DuckDB, explicit point-in-time replay, stable IDs, neutral unknown handling, chronological folds, series-clustered intervals, native CatBoost, the fixed Elo ensemble, versioned artifacts, and a single training and prediction feature path.
It also preserves the old v3 candidates as exact 143-feature matched controls.

It deliberately discards hard-coded team-to-region mappings, same-patch aggregates, final historical ratings, random splits, post-target updates, separate side scalers, automatic tier expansion, weight or K-factor searches, and accuracy-only selection.
It also rejects the raw metric winner when its league gate fails.
No neural or Siamese model was added because the auditable regional candidate still has unresolved calibration and robustness risks.

## Verification

The complete suite passes 102 tests.
Statement coverage is 89.79 percent against the 85 percent CI floor.
Ruff formatting and linting, strict mypy checking, lock validation, repository hygiene, credential-pattern scanning, source and wheel builds, artifact checksums, old-artifact loading, and separate-process prediction pass.

The backtest report SHA-256 is `932aac8448e6d33044e1d206d2c1351f7d2a2bb42d406f58cadca668a9e9ba6a`.
The confirmation report SHA-256 is `c734284d4c9db462b92ac6b5ed6377c6ce33696c0330cc7e3c24d78b4fb734b6`.
The development refit report SHA-256 is `3179f8579dd815626ead34cdace9cc91e4a84a62e38d038fbeb899dd58af6960`.

## Exact commands

Install the locked environment:

```bash
uv sync --locked --all-groups
```

Build a fresh analytical database from the pinned snapshot:

```bash
uv run lolpredictor ingest-leaguepedia \
  --database var/v4-regional/matches.duckdb \
  --input data/raw/leaguepedia-2020-2026-through-20260730T234000Z-v2.json
```

Generate the point-in-time v5 feature table:

```bash
uv run lolpredictor features \
  --database var/v4-regional/matches.duckdb \
  --config configs/v4-regional-development.yaml
```

Run the pre-2026 rolling comparison:

```bash
uv run lolpredictor backtest \
  --database var/v4-regional/matches.duckdb \
  --config configs/v4-regional-development.yaml \
  --reports var/v4-regional/reports
```

Evaluate only the frozen candidate on the already-opened diagnostic:

```bash
uv run lolpredictor confirm-selection \
  --database var/v4-regional/matches.duckdb \
  --config configs/v4-regional-development.yaml \
  --backtest-report var/v4-regional/reports/backtest-report.json \
  --output var/v4-regional/reports/confirmation-report.json
```

Build the warned shadow artifact:

```bash
uv run lolpredictor refit-development \
  --database var/v4-regional/matches.duckdb \
  --config configs/v4-regional-development.yaml \
  --backtest-report var/v4-regional/reports/backtest-report.json \
  --confirmation-report var/v4-regional/reports/confirmation-report.json \
  --registry var/v4-regional/artifacts \
  --reports var/v4-regional/reports
```

Run a structured prediction:

```bash
uv run lolpredictor predict \
  --artifact var/v4-regional/artifacts/leaguepedia-2020-2026-v4-regional-development-elo_catboost_regional_raw_blend_50-20260731T032543Z-3dc6f839 \
  --input examples/lck_hypothetical_draft.json \
  --output var/v4-regional/reports/sample-shadow-prediction.json
```

Run all quality checks:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv build
uv run pytest --cov=src/lolpredictor --cov-report=term-missing --cov-fail-under=85
```

## Remaining risks and next milestone

The model remains a shadow model because the opened 2026 ECE is 0.063098 and no untouched post-July promotion interval exists yet.
The regional gain is concentrated in international events, while several domestic breakdowns regress slightly.
The source changes region taxonomy over time, and no validated versioned mapping yet connects Europe to EMEA, older regional labels to Americas, or older Pacific labels to Asia Pacific.
The eligible population has no tournament-tier variation, so tier effects remain unidentified.
LCS robustness remains close to the fixed regression boundary.
The exact-draft contribution remains smaller and less stable than team and roster strength.

The next highest-value model milestone is a preregistered dynamic Bradley-Terry or state-space strength model with partial pooling across team, roster, league, and region.
It should model strength uncertainty and seasonal drift explicitly, retain outcome updates after timestamp groups, and enter the same rolling comparison as a separate auditable candidate.
No post-July 2026 outcomes should be opened until that protocol and every promotion gate are frozen.
Promotion still requires at least 750 genuinely new games from at least 300 series.

The next screenshot milestone remains a rights-cleared, recording-grouped target-overlay corpus with at least 100 verified frames from at least three recordings and ten matches.
Parser precision and probability quality must retain separate release gates.

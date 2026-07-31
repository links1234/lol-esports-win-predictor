# V6 hierarchical state-space results

## Status and decision

The preregistered v6 hierarchical state-space study is complete.
The real run finished in 161.55 seconds.

V6 produced the strongest nested historical point estimate in the project so far.
Its outer selection policy scored 0.606542 log loss, compared with 0.609566 for the fixed v4 regional control.
The absolute improvement is 0.003024 log loss.

The improvement is promising but does not pass the frozen replacement gate.
The series-clustered paired 95 percent interval for v6-minus-v4 is -0.006905 to 0.000738, so its upper bound remains above zero.
The League of Legends Championship Series slice regressed by 0.010576, narrowly exceeding the frozen 0.01 league limit.

The correct recommendation is therefore to retain v4 as the current shadow model.
The locked v6 blend is available as a development-only challenger for prospective shadow comparison.
Neither model is production-promoted.

## Frozen boundary

The study followed [`docs/v6-state-space-protocol.md`](v6-state-space-protocol.md).
The exclusive supervised cutoff was `2026-01-01T00:00:00Z`.
DuckDB returned only matches strictly earlier than that timestamp.

The bounded source contained 84,862 matches from January 3, 2020 through December 29, 2025.
The official `Primary` or `Premier` modeling population contained 20,380 games.
The four outer score intervals contained 8,135 games from 3,582 series.
Their combined interval ran from March 21, 2023 through November 9, 2025.

No 2026 result influenced a latent parameter used in evaluation, fitted coefficient, mapper, candidate selection, stopping decision, or interpretation.
The later development refit uses already-opened 2026 outcomes only after the finalist was immutable.

## Model implementation

The model represents global side, league side, region, team, player, exact roster, role champion, patch-role-champion, and role matchup effects as diagonal Gaussian latent components.
Each component stores a mean, variance, observation count, and last-observed timestamp.
Uncertainty increases during inactivity and the mean gradually returns toward its centered prior.

Every match at one timestamp reads the same projected state.
Gradients and diagonal curvature are accumulated for the complete timestamp group before any component is updated.
This makes simultaneous outcomes isolated and the final state independent of row ordering inside the group.

The replay created 23,004 observed latent components.

| Component | Count |
| --- | ---: |
| Global side | 1 |
| League side | 22 |
| Region | 14 |
| Team | 246 |
| Player | 2,081 |
| Exact roster | 2,363 |
| Role champion | 472 |
| Patch-role-champion | 12,761 |
| Role matchup | 5,044 |

The feature contract contains 172 numeric fields, including 17 state-space fields.
Historical rows and live requests use the same computation path.
The old v4 control remains pinned to its original 155-field contract and cannot consume the new fields.

## Pooled outer comparison

Log loss is the primary metric.

| Candidate or policy | Log loss | Brier | ECE | ROC-AUC | Accuracy | Games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V6 nested selection policy | **0.606542** | **0.209625** | 0.011403 | **0.729709** | **66.64%** | 8,135 |
| Fixed v4 regional control | 0.609566 | 0.211037 | 0.018571 | 0.725704 | 66.27% | 8,135 |
| Team-roster logistic control | 0.611943 | 0.212361 | **0.009626** | 0.719953 | 65.42% | 8,135 |
| Elo-only control | 0.620256 | 0.215778 | 0.019824 | 0.711530 | 66.01% | 8,135 |

V6 improved Elo by 0.013715 log loss.
Its paired interval against Elo is -0.018545 to -0.008755 and stays below zero.

V6 improved team-roster logistic by 0.005401.
Its paired interval against team-roster logistic is -0.009207 to -0.001540 and also stays below zero.

V6 improved v4 by 0.003024.
Its paired interval against v4 is -0.006905 to 0.000738 and crosses zero.

Accuracy, Brier score, ECE, and ROC-AUC all moved in a favorable direction versus v4.
Those secondary metrics do not override the failed paired and league gates.

## Outer selections

Each outer candidate was chosen from its own inner folds before that outer outcome was touched.

| Outer fold | Inner-selected candidate | Outer log loss | V4 log loss | Difference | Games |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | State-space augmented logistic | 0.610341 | 0.610498 | -0.000157 | 2,048 |
| 2 | State-space augmented logistic | 0.591660 | 0.603661 | -0.012001 | 2,030 |
| 3 | State-space augmented logistic | 0.603686 | 0.599870 | +0.003816 | 2,034 |
| 4 | 70% v4 plus 30% state-space logistic | 0.620499 | 0.624296 | -0.003796 | 2,023 |

The augmented logistic challenger was selected in the first three outer folds.
The conservative 30 percent state-space blend was selected in the fourth.
No outer fold used the fallback.

The policy improved v4 in three of four folds.
The third fold regression stayed below the frozen 0.01 fold limit.
The second fold supplied most of the pooled gain, which is one reason the clustered uncertainty interval remains important.

## Locked challenger

The fourth outer fold's inner evidence locked `state_space_v4_blend_30`.
Its lock fingerprint is `04e267197e08086fb393e06528c1a5e8a9e1bae8f63b5566080402b98a9b7d06`.

The locked model assigns 70 percent weight to the unchanged v4 probability and 30 percent weight to the state-space augmented logistic probability.
The augmented component uses L2 logistic regression with `C=0.10`.
It consumes core, team-strength, roster-form, draft-interaction, regional, and state-space numeric groups.
It does not consume sparse categorical identities or the old independent champion-rate group.

The locked blend scored 0.620499 log loss on its once-opened fourth outer interval.
V4 scored 0.624296 on the same 2,023 games.
The locked blend also improved fourth-outer Brier score from 0.217703 to 0.216109, ECE from 0.028083 to 0.012227, ROC-AUC from 0.703900 to 0.708280, and accuracy from 64.36 percent to 64.76 percent.

The raw state-space probability and its Platt-mapped version were not eligible inner winners.
This indicates that the latent filter is currently more useful as structured evidence inside a regularized model than as a complete standalone forecast.

## What the fitted model used

The development refit's largest standardized augmented-logistic coefficients included roster experience, state-space team strength, state-space total log odds, state-space regional strength, and state-space player strength.
The largest state-space coefficients were:

| State-space field | Standardized coefficient |
| --- | ---: |
| Team strength difference | +0.2883 |
| Total latent log-odds mean | +0.2695 |
| Region strength difference | +0.1402 |
| Player strength difference | +0.1033 |
| Role matchup strength | +0.0493 |
| Raw state-space probability | +0.0402 |
| Role champion strength difference | +0.0301 |
| Patch-role-champion difference | +0.0300 |

These coefficients are descriptive rather than causal.
Several strength features are correlated, so an individual coefficient must not be interpreted as an isolated effect size.
The useful conclusion is that team, region, player, and draft matchup state all survived regularization with nontrivial weight.

## League and temporal robustness

The policy improved several large groups.

| Group | Games | V6-minus-v4 log loss |
| --- | ---: | ---: |
| World Championship | 365 | -0.032290 |
| Liga Latinoamerica | 334 | -0.019247 |
| Vietnam Championship Series | 539 | -0.011840 |
| Circuit Brazilian League of Legends | 415 | -0.011699 |
| LoL Champions Korea | 1,307 | -0.005079 |
| LoL EMEA Championship | 775 | -0.003676 |

It regressed in several other groups.

| Group | Games | V6-minus-v4 log loss |
| --- | ---: | ---: |
| Mid-Season Invitational | 234 | +0.013862 |
| League of Legends Championship Series | 363 | +0.010576 |
| Tencent LoL Pro League | 1,980 | +0.003981 |
| Pacific Championship Series | 498 | +0.003918 |

The LCS regression failed the fixed major-league gate by 0.000576.
The March 2025 monthly slice also regressed sharply, but it contains only 124 games and should be treated as a drift warning rather than a stable universal estimate.
The mixed temporal pattern supports prospective shadowing instead of immediate replacement.

## Frozen gate outcome

| Gate | Result |
| --- | --- |
| At least 750 outer games | Pass |
| At least 300 outer series | Pass |
| At least 0.003 log-loss improvement over v4 | Pass at 0.003024 |
| Paired interval upper bound below zero | Fail at +0.000738 |
| ECE at most 0.04 | Pass at 0.011403 |
| Every outer regression at most 0.01 | Pass |
| Every major-league regression at most 0.01 | Fail for LCS at +0.010576 |

The overall gate failed.
V4 remains the recommended shadow model.

## Development artifact

The immutable finalist was refit after the study on all currently available outcomes.
This estimates current coefficients and state but does not create new evaluation evidence.

The artifact is:

```text
var/v6-state-space/artifacts/leaguepedia-pre2026-v6-state-space-development-state_space_v4_blend_30-20260731T082553Z-f108f0a1
```

Its model version is `v6-state-space-locked+cfg.f108f0a1.data.f4e69cdb`.
Its feature-state cutoff is July 30, 2026 at 19:52 UTC.
The refit source contains 93,041 accepted games and 22,162 official `Primary` or `Premier` modeling games.
The artifact is marked `development`, records that prospective promotion failed, embeds the complete lock, and warns on every prediction.

A separate process loaded the artifact and predicted the saved hypothetical LCK draft.
It returned 67.4152 percent blue and 32.5848 percent red.
The v4 artifact returned 67.4466 percent blue on the same request.
Those close sample probabilities are expected because the locked challenger retains 70 percent v4 weight.

## Preserved and deliberately discarded

V6 preserves DuckDB, strict source validation, exact timestamp replay, simultaneous-match isolation, chronological nested evaluation, fixed controls, log-loss selection, clustered uncertainty, shared training and prediction features, configuration locks, checksum-verified artifacts, JSON and screenshot interfaces, and explicit development warnings.

V6 adds uncertainty-aware latent means and variances, inactivity drift, exact roster transitions, additive region and player effects, hierarchical global and patch-local champion effects, antisymmetric role matchups, order-independent batch curvature updates, a fixed v4 feature boundary, lock validation, and uncertainty and coverage warnings.

V6 deliberately avoids random splits, final historical ratings, full-patch aggregates, post-2025 selection, unrestricted hyperparameter search, unpooled identity coefficients, separate side scalers, accuracy-only claims, and neural escalation.
It also refuses to replace v4 based on a favorable point estimate when the preregistered uncertainty and league gates fail.

## Verification and fingerprints

The synthetic v6 workflow builds a fixture database, replays latent state, evaluates nested folds, writes a lock, refits an artifact, loads it in a separate process, and predicts a later draft.
Unit tests cover uncertainty contraction, inactivity expansion, strict projection time, order-independent batches, target and simultaneous leakage, serialization, frozen configuration, and shared artifact prediction.
Legacy v4 and v5 artifacts remain loadable and reproduce their earlier sample probabilities exactly.
The complete suite passes 124 tests with 90.20 percent statement coverage.
Formatting, linting, strict type checking, dependency-lock validation, source distribution construction, and wheel construction also pass.
The disabled state-space configuration remains absent from resolved older experiments, preserving the exact v4 configuration fingerprint `3dc6f839c7c4403ec0eaafcacf9917aec5548f599d13eedd7c4f2bedb4e66407`.
The enabled v6 experiment retains its locked configuration fingerprint.

The study-report SHA-256 is `50e20a2d43a75ad3144a884e3ada0a9e12ff6dee8697a575ef4270f7c473b778`.
The locked-finalist SHA-256 is `0535d100cca5de27f4975c250d6d923ee6345e00b5241534641a8c2dead247a9`.
The development-refit report SHA-256 is `39b12c0a245f19e5979fe854360222cec913e66b0f1fbf039b1ec1f634c18b64`.
The sample-prediction SHA-256 is `bd6b42b72846859cd6b9d6166fdf09ff942fe5a9d2528eab92e2e37bcd386a46`.

## Exact commands

Install the locked environment:

```bash
uv sync --locked --all-groups
```

Run the real pre-2026 v6 comparison:

```bash
uv run lolpredictor state-space-study \
  --database var/v4-regional/matches.duckdb \
  --config configs/v6-state-space.yaml \
  --study var/v6-state-space
```

The command refuses to overwrite an existing study.
Use a new study directory for an independent rerun.

Refit the immutable finalist as a warned development artifact:

```bash
uv run lolpredictor state-space-refit \
  --database var/v4-regional/matches.duckdb \
  --config configs/v6-state-space.yaml \
  --locked-finalist var/v6-state-space/locked-finalist.json \
  --study-report var/v6-state-space/study-report.json \
  --registry var/v6-state-space/artifacts \
  --output var/v6-state-space/refit-report.json
```

Predict structured JSON:

```bash
uv run lolpredictor predict \
  --artifact var/v6-state-space/artifacts/leaguepedia-pre2026-v6-state-space-development-state_space_v4_blend_30-20260731T082553Z-f108f0a1 \
  --input examples/lck_hypothetical_draft.json \
  --output var/v6-state-space/sample-shadow-prediction.json
```

Run the synthetic v6 end-to-end test:

```bash
uv run pytest tests/e2e/test_state_space_workflow.py
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

The paired interval still permits a small regression versus v4.
The LCS, MSI, LPL, and March 2025 slices show that competition transitions and regional drift are not fully controlled.
The diagonal approximation ignores posterior covariance among team, player, roster, and champion effects.
Exact patch deviations remain sparse, and the large serialized state is approximately 50 MB in JSON form.
No untouched post-July 2026 interval exists, so neither v4 nor v6 can be promoted prospectively.

The highest-value next milestone is prospective dual-model shadow logging.
Every new confirmed draft should record immutable v4 and v6 probabilities before match start, then attach the result only after the game ends.
Promotion should be reconsidered only after at least 750 new games from at least 300 series satisfy the same log-loss, calibration, paired-interval, and major-league gates.

The highest-value modeling follow-up is a preregistered drift-aware regional mixture that addresses the identified LCS, MSI, and LPL instability without tuning on 2026 outcomes.
It should be attempted only with nested evaluation and should remain secondary to collecting genuinely new prospective evidence.

The screenshot milestone remains independent.
A rights-cleared, recording-grouped real corpus still needs to pass extraction precision, exact-draft accuracy, and safe-abstention gates before image input is unattended.

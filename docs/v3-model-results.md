# V3 model research and results

## Status

This report preserves the initial v3 experiment and has been followed by `docs/v3-calibration-results.md`.
V3 now has a complete leakage-safe development comparison, a locked confirmation diagnostic, and a separate-process shadow prediction artifact.
The selected shadow model is a 50 percent Elo and 50 percent native-probability numeric CatBoost ensemble.
It is better than Elo on the available chronological evidence, but it is not a promoted production model.
The next unbiased promotion holdout starts after July 30, 2026.

The development artifact is intentionally labeled `artifact_purpose: development`.
Every prediction from it warns that the prospective promotion gate has not passed.

## Competitive research conclusions

Riot's 2026 First Selection rules separate map side from first versus second draft pick.
The model therefore stores and models those fields independently.
Source: [Riot Games, Season Start 2026](https://lolesports.com/en-PH/news/season-start-2026-lol-esports).

Riot's Fearless Draft rules make earlier picks in the same series part of the next game's pregame state.
Series score, game number, and explicit Fearless exclusions are valid inputs, but the current source does not identify Fearless exclusions reliably enough to reconstruct them automatically.
Sources: [Riot Games, LoL Esports in 2025](https://lolesports.com/en-GB/news/lol-esports-in-2025) and [Riot Games, Fearless Draft Takes Over 2025](https://lolesports.com/en-PH/news/fearless-draft-takes-over-2025).

Published League of Legends research supports separating player skill, champion strength, and player-champion familiarity.
It also supports testing team interactions and role-aware histories.
Sources: [Chen et al., Player Skill Decomposition](https://arxiv.org/abs/1702.06253), [Do et al., Player-Champion Experience](https://arxiv.org/abs/2108.02799), and [Lee et al., DraftRec](https://arxiv.org/abs/2204.12750).

PandaSkill identifies isolated regional rating pools and role-specific player performance as important professional-scene problems.
Its postgame performance inputs may only update later matches and require a separately validated source before they can be added here.
Source: [De Bois et al., PandaSkill](https://arxiv.org/abs/2501.10049).

The research justified the tested feature families, but the ablations determine whether they help this professional dataset.
The expanded role-aware linear and histogram-tree models did not beat their smaller matched controls.
This is why the current selection uses a regularized ensemble rather than a neural model or an unrestricted identity model.

## Corrected source

The immutable snapshot is `data/raw/leaguepedia-2020-2026-through-20260730T234000Z-v2.json`.
It was retrieved at `2026-07-30T23:42:50+00:00` for the half-open interval ending at `2026-07-30T23:40:00+00:00`.
Its SHA-256 is `c0f94d5982d7f8ddb82ea7c3de6203a5d19232ba3a13157f9ce927dbcedf9db0`.

The snapshot contains 95,588 source rows.
Strict validation accepted 93,041 games and quarantined 2,547 games.
The accepted history runs from January 3, 2020 through July 30, 2026 at 19:52 UTC.
The official Primary or Premier supervised population contains 22,162 games.
All 93,041 accepted games remain available for point-in-time state replay.

The source audit found that direct CargoExport serialization collapsed numeric-looking patch strings.
For example, `25.10` became `25.1` and `25.20` became `25.2`.
The v2 fetch query forces patch to a string with a sentinel before JSON serialization, and ingestion removes the sentinel after validation.
Regression tests prove that `25.10`, `25.20`, and `26.10` remain distinct.

The corrected database uses schema version 3.
The historical feature table uses `point-in-time-v4`.

## Leakage-safe protocol

Every historical feature row is computed before the target outcome updates state.
Matches with the same timestamp are computed as one batch before any outcome in that batch is applied.
Team Elo, player Elo, form, champion statistics, role statistics, synergy, matchup statistics, player-champion history, roster continuity, and all coverage counts obey this boundary.

The supervised population filter uses only structural pregame fields.
Tests flip every target outcome and prove that the selected match IDs do not change.

Preprocessing is fitted only on each fold's fit interval.
Probability calibration uses only the trailing calibration interval.
Recency weights use only the maximum timestamp inside the fit interval.
CatBoost receives chronologically ordered rows with `has_time=True`.
Unseen validation categories must still produce finite bounded probabilities.

Automated tests also cover future-label perturbation, simultaneous-match isolation, provenance timestamps, historical-versus-live feature parity, patch identity, and separate-process artifact loading.

Candidate development used three expanding rolling-origin folds.
The 6,087 combined rolling test games belong to 2,638 series.
The final development target was played on November 9, 2025.
January through March 2026 was reserved until the candidate and robustness rule were locked.
April through July 2026 was not evaluated by the v3 workflow.

## Models compared

The table is sorted by aggregate rolling test log loss.
Accuracy is reported for context and was not used for selection.

| Model | Log loss | Brier | ECE | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| 25% Elo + 75% native numeric CatBoost | 0.609901 | 0.211270 | 0.022622 | 0.724947 | 66.27% |
| Native numeric CatBoost | 0.610922 | 0.211728 | 0.016061 | 0.723241 | 66.32% |
| 50% Elo + 50% native numeric CatBoost | 0.611057 | 0.211768 | 0.028999 | 0.723814 | 66.42% |
| Platt-calibrated numeric CatBoost | 0.611619 | 0.212009 | 0.019532 | 0.722463 | 66.09% |
| Native legacy-feature CatBoost | 0.611888 | 0.212100 | 0.015683 | 0.721921 | 66.29% |
| Team-roster logistic | 0.612719 | 0.212665 | 0.009380 | 0.719184 | 65.65% |
| Platt-calibrated legacy-feature CatBoost | 0.612806 | 0.212475 | 0.014088 | 0.720765 | 66.26% |
| Legacy-feature histogram trees | 0.613549 | 0.212859 | 0.014819 | 0.719763 | 66.19% |
| Role-expanded histogram trees | 0.614161 | 0.213239 | 0.017976 | 0.718432 | 66.27% |
| Native team-roster CatBoost | 0.615134 | 0.213590 | 0.014218 | 0.717595 | 65.94% |
| Platt-calibrated team-roster CatBoost | 0.615972 | 0.213916 | 0.015724 | 0.716729 | 66.06% |
| Team-roster histogram trees | 0.616855 | 0.214131 | 0.014770 | 0.716572 | 65.96% |
| Elo only | 0.619977 | 0.215634 | 0.021001 | 0.712263 | 65.85% |
| Elo-logistic blend | 0.620637 | 0.216095 | 0.026972 | 0.711198 | 65.62% |
| Numeric plus raw identity-context CatBoost | 0.624189 | 0.217106 | 0.012638 | 0.707381 | 64.91% |
| Legacy combined logistic | 0.626153 | 0.217935 | 0.041278 | 0.709911 | 65.45% |
| Role-expanded combined logistic | 0.637219 | 0.222327 | 0.045695 | 0.698674 | 64.38% |
| Recency-weighted logistic | 0.689410 | 0.239607 | 0.092881 | 0.671947 | 63.45% |
| Blue-side base rate | 0.691354 | 0.249104 | 0.005281 | 0.498837 | 53.05% |
| Draft-only logistic | 0.695700 | 0.250977 | 0.036365 | 0.539298 | 54.59% |

Every row contains 6,087 rolling test games.

## Selection decision

The 25 percent Elo blend has the lowest aggregate log loss.
It was rejected by the preregistered robustness rule because its LCS log loss regressed by 0.012943 against Elo, above the 0.01 limit.

The selected 50 percent Elo blend improves Elo in every rolling fold by 0.012881, 0.011650, and 0.002201 log loss.
Its aggregate log loss is 0.611057 versus 0.619977 for Elo.
Its aggregate Brier score is 0.211768, ECE is 0.028999, ROC-AUC is 0.723814, and accuracy is 66.42 percent.
Its series-clustered 95 percent log-loss interval is 0.602486 to 0.620696.
Its paired candidate-minus-Elo interval is -0.011501 to -0.006433.

The best simple control is team-roster logistic at 0.612719 log loss.
The selected blend's paired interval against that control is -0.004773 to 0.001638.
That interval crosses zero, so the current data does not establish that the blend is better than the simple control even though its point estimate is lower.

The selected blend stays within the 0.01 major-league regression limit for every configured league with at least 100 test games.
Its largest regression is 0.006663 in the 192-game LCS slice.

## Locked 2026 confirmation

After selection was locked, the model was fitted only through November 9, 2025 and evaluated on 744 games from January through March 2026.
This interval had already been inspected during v2 work, so it is an opened diagnostic and not promotion evidence.

The selected blend scored 0.668306 log loss versus 0.695109 for Elo.
Its paired series-clustered candidate-minus-Elo interval is -0.037356 to -0.016362.
It improved log loss over Elo in every represented league.
Its Brier score is 0.237686, ROC-AUC is 0.640003, and accuracy is 58.33 percent.

Its ECE is 0.063239, which fails the 0.04 calibration target.
This result supports the model's ranking signal but also shows that probability calibration shifted in the 2026 competitive regime.
No production-quality claim is justified yet.

## Shadow artifact

The current local shadow artifact is:

```text
var/v3-development/artifacts/leaguepedia-2020-2026-v3-development-development-elo_catboost_numeric_blend_50-20260731T002919Z-9cd555b5
```

Its model version is `v3-development-20260730+cfg.9cd555b5.data.f4e69cdb`.
Its feature-state data cutoff is `2026-07-30T19:52:00+00:00`.
The supervised estimator uses 19,953 fit games and a trailing 2,209-game calibration interval.
The artifact contains a calibrated Elo component and a native-probability CatBoost component at equal weights.

The artifact is for shadow use only.
It has not passed the prospective release gate.

## Exact commands

Install and verify dependencies:

```bash
uv sync --locked --all-groups
```

Fetch the corrected immutable source:

```bash
uv run lolpredictor fetch-leaguepedia \
  --output data/raw/leaguepedia-2020-2026-through-20260730T234000Z-v2.json \
  --start 2020-01-01T00:00:00+00:00 \
  --end 2026-07-30T23:40:00+00:00 \
  --retrieved-at 2026-07-30T23:42:50+00:00 \
  --page-size 5000
```

Build the corrected DuckDB database:

```bash
uv run lolpredictor ingest-leaguepedia \
  --database var/v3-development/matches.duckdb \
  --input data/raw/leaguepedia-2020-2026-through-20260730T234000Z-v2.json
```

Generate point-in-time features:

```bash
uv run lolpredictor features \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-development.yaml
```

Train and compare every development candidate:

```bash
uv run lolpredictor backtest \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-development.yaml \
  --reports var/v3-development/reports
```

Evaluate only the locked candidate on the opened validation diagnostic:

```bash
uv run lolpredictor confirm-selection \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-development.yaml \
  --backtest-report var/v3-development/reports/backtest-report.json \
  --output var/v3-development/reports/confirmation-report.json
```

Refit the locked family as a non-promoted shadow artifact:

```bash
uv run lolpredictor refit-development \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-development.yaml \
  --backtest-report var/v3-development/reports/backtest-report.json \
  --confirmation-report var/v3-development/reports/confirmation-report.json \
  --registry var/v3-development/artifacts \
  --reports var/v3-development/reports
```

Run a separate-process prediction:

```bash
uv run lolpredictor predict \
  --artifact var/v3-development/artifacts/leaguepedia-2020-2026-v3-development-development-elo_catboost_numeric_blend_50-20260731T002919Z-9cd555b5 \
  --input examples/lck_hypothetical_draft.json \
  --output var/v3-development/reports/sample-shadow-prediction.json
```

Run all engineering checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=src/lolpredictor --cov-report=term-missing --cov-fail-under=80
uv build
```

Run the complete synthetic CI workflow:

```bash
uv run lolpredictor run-all \
  --workdir var/fixture-v3-run \
  --config configs/baselines.yaml \
  --sample examples/sample_draft.json
```

## Preserved and discarded ideas

The rebuild preserves DuckDB, explicit point-in-time transformations, configuration-driven experiments, team and player ratings, role-aligned drafts, batch and live interfaces, and versioned artifacts.
It also preserves the useful legacy insight that team strength and roster context are usually stronger than draft-only statistics.

The rebuild rejects random splits, final Elo reused for history, same-patch full-dataset aggregates, separate team scalers, hard-coded identity mappings, unsafe evaluation, committed databases and models, accuracy-only claims, and neural-first development.
It also rejects the assumption that more raw identities or more role features must improve future performance.
Both assumptions lost their matched temporal ablations.

## Remaining risks and next milestone

The largest current risk is 2026 calibration shift.
The follow-up experiment compared native, Platt, monotone beta, and isotonic mappings inside nested chronological folds.
Native output retained the best eligible log loss, while the opened 2026 diagnostic still failed the calibration target.
The full follow-up is recorded in `docs/v3-calibration-results.md`.

First-pick coverage is sparse before 2026, and automatic Fearless exclusions are absent.
The ingestion layer needs an explicit, versioned tournament-rule source before those fields can be reconstructed safely.

The next high-value predictive feature is a prior-match-only role performance rating from a licensed public source.
It should use postgame performance from earlier matches only, normalize within role and patch, and update a regional plus global rating before the next target.
This directly tests the PandaSkill hypothesis without leaking the target game.

Regional meta-ratings and uncertainty-aware player ratings are also promising because domestic rating pools interact only at international events.
They should be added as matched ablations against the current ensemble.

The next unbiased promotion holdout starts after July 30, 2026.
It must remain unopened until at least 750 games from 300 series are available and the model, calibration method, feature contract, and gates are frozen.

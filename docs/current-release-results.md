# Current release results

## Outcome

The July 30, 2026 probability-model release is complete.
The combined logistic candidate passed every aggregate, calibration, uncertainty, and sample-size gate but failed the fixed LPL robustness gate.
The production artifact therefore uses the preregistered Elo fallback.
The stronger combined logistic artifact remains an evaluation artifact and is not the promoted production model.

## Source and cutoff

The source is a pinned [Leaguepedia Cargo export](https://lol.fandom.com/wiki/Special:CargoExport) joined through `ScoreboardGames`, `MatchScheduleGame`, and `Tournaments`.
The [ScoreboardGames declaration](https://lol.fandom.com/wiki/Module%3ACargoDeclare/ScoreboardGames) documents the game join key and role-ordered pick arrays.
The [Tournaments declaration](https://lol.fandom.com/wiki/Module%3ACargoDeclare/Tournaments) documents the tournament join key and canonical league field.

The extraction interval is January 1, 2020 at 00:00 UTC through July 30, 2026 at 19:00 UTC, with the end exclusive.
The pinned retrieval timestamp is July 30, 2026 at 20:35:32 UTC.
The 132 MB snapshot contains 95,579 source games.
Its SHA-256 is `d24db326229702e86a47bf0ec63ec70e9efd59a1aec74a70fd7442d176795559`.

Canonical ingestion accepted 93,032 games and quarantined 2,547 games.
The accepted dataset fingerprint under the current canonical schema is `c6c74cf69dcad65a097304b8f9d9c0dfb9ddbe6900c0cddc229c424541067b30`.
The accepted feature history ends on July 30, 2026 at 18:12 UTC.

| Quarantine reason | Games |
| --- | ---: |
| Missing required patch, side, or league | 2,008 |
| Invalid series sequence | 194 |
| Invalid series timestamps | 168 |
| Contradictory canonical draft | 110 |
| Inconsistent side assignment | 67 |

The adapter preserves Riot's real `25.S1.1` through `25.S1.3` patch identifiers.
Riot's [25.04 patch notes](https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-04-notes/) explain the temporary seasonal naming and the return to year-and-number naming.
Unavailable Leaguepedia ban slots represented as `None` are omitted because bans are optional.
Actual duplicate picks, duplicate champion bans, ban-pick overlaps, missing patch, and missing side remain quarantined.

The accepted 2026 corpus contains 8,170 games.
First-pick is known for 28.49 percent of those games.
It is known for 935 of 2,913 validation games and 1,393 of 5,257 final-holdout games.
Missing first-pick values were not inferred from map side.

## Temporal boundaries

Estimator development ends before January 1, 2026.
Candidate selection uses January 1 through March 31, 2026.
The selected family is refitted on all 87,775 pre-April games with a trailing chronological calibration interval.
Its evaluation feature-state cutoff is March 31, 2026 at 23:12 UTC.

The final holdout contains 5,257 games across 2,569 series from April 1 through July 30, 2026.
The holdout has now been opened and cannot be reused as an untouched gate for any changed model, feature, source filter, or threshold.

Every real feature row passed the strict provenance check.
There were zero cases where Elo, form, champion, player-champion, or general feature history reached or exceeded the target match timestamp.

## Rolling development results

The rolling-origin report contains 25,517 pre-2026 out-of-time predictions over three folds.

| Candidate | Log loss | Brier | ECE | ROC-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blue-side base rate | 0.6914 | 0.2491 | 0.0021 | 0.4968 | 52.98% |
| Elo only | 0.6428 | 0.2260 | 0.0089 | 0.6743 | 62.68% |
| Team and roster logistic | 0.6209 | 0.2162 | 0.0142 | 0.7086 | 65.06% |
| Draft-only logistic | 0.6867 | 0.2468 | 0.0119 | 0.5583 | 54.67% |
| Combined logistic | 0.6185 | 0.2151 | 0.0141 | 0.7122 | 65.40% |
| Gradient-boosted trees | **0.6170** | **0.2145** | 0.0071 | **0.7135** | 65.38% |
| 30% Elo plus 70% combined logistic | 0.6196 | 0.2156 | 0.0105 | 0.7104 | 65.27% |

The release-selected combined logistic family improves rolling log loss over Elo by 0.0243.

## Validation and final holdout

Candidate selection used validation log loss only.
Combined logistic won validation at 0.6044, narrowly ahead of gradient boosting at 0.6050.
The holdout column uses models refitted only through March 31.

| Candidate | Validation log loss | Holdout log loss | Holdout Brier | Holdout ECE | Holdout ROC-AUC | Holdout accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blue-side base rate | 0.6933 | 0.6909 | 0.2489 | 0.0046 | 0.5000 | 53.40% |
| Elo only | 0.6571 | 0.6340 | 0.2218 | 0.0200 | 0.6901 | 63.88% |
| Team and roster logistic | 0.6083 | 0.6043 | 0.2086 | 0.0143 | 0.7316 | 67.17% |
| Draft-only logistic | 0.6835 | 0.6852 | 0.2461 | 0.0176 | 0.5658 | 55.41% |
| Combined logistic | **0.6044** | 0.6005 | 0.2069 | 0.0140 | 0.7363 | **67.57%** |
| Gradient-boosted trees | 0.6050 | **0.6001** | **0.2066** | **0.0134** | **0.7375** | 67.28% |
| 30% Elo plus 70% combined logistic | 0.6110 | 0.6036 | 0.2081 | 0.0250 | 0.7352 | 67.57% |

The combined logistic holdout improvement over Elo is 0.0335 log loss.
Its series-clustered paired 95 percent interval for selected-minus-Elo log loss is -0.0407 to -0.0264.
Its holdout Brier score and calibration error both improve on Elo.

The fixed major-league gate failed because combined logistic regressed against Elo by 0.0167 log loss on 272 LPL games.
The maximum allowed regression was 0.01.
LEC regressed by 0.0070 and passed, while LCK, LCS, CBLOL, and LCP improved or remained within the limit.

The promotion report therefore records `promotion_passed: false` and `recommended_candidate: elo_only`.
Accuracy did not override the failed league log-loss gate.

## Production artifact and sample

The production Elo artifact is refit on all 93,032 accepted games.
Its data cutoff is July 30, 2026 at 18:12 UTC.
The refit command writes a timestamped artifact directory under the configured registry, for example:

```text
var/current-release/production-artifacts/leaguepedia-2020-2026-current-release-production-elo_only-<timestamp>-<config-hash>
```

The illustrative current LCK JSON produced:

```json
{
  "blue_win_probability": 0.6419212836469772,
  "red_win_probability": 0.3580787163530228,
  "model_version": "current-2026-release-v1+cfg.<config-hash>.data.c6c74cf6",
  "data_cutoff_timestamp": "2026-07-30T18:12:00Z",
  "warnings": [
    "Unknown home regions: Hanwha Life Esports, Dplus Kia; using neutral regional priors"
  ]
}
```

This production fallback is team-strength sensitive but deliberately not draft sensitive.
The combined logistic evaluation artifact is draft sensitive, but it is not promoted because of the LPL failure.

## Exact reproduction commands

Install and verify the locked environment:

```bash
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=src/lolpredictor --cov-report=term-missing --cov-fail-under=80
uv build
```

Fetch and ingest the pinned source:

```bash
uv run lolpredictor fetch-leaguepedia \
  --output data/raw/leaguepedia-2020-2026-through-20260730T190000Z.json \
  --start 2020-01-01T00:00:00+00:00 \
  --end 2026-07-30T19:00:00+00:00 \
  --retrieved-at 2026-07-30T20:35:32+00:00 \
  --page-size 5000

sha256sum data/raw/leaguepedia-2020-2026-through-20260730T190000Z.json

uv run lolpredictor ingest-leaguepedia \
  --database var/current-release-v2/matches.duckdb \
  --input data/raw/leaguepedia-2020-2026-through-20260730T190000Z.json
```

Generate features, backtest, select, and evaluate:

```bash
uv run lolpredictor features \
  --database var/current-release-v2/matches.duckdb \
  --config configs/current-release.yaml

uv run lolpredictor backtest \
  --database var/current-release-v2/matches.duckdb \
  --config configs/current-release.yaml \
  --reports var/current-release-v2/backtest-reports

uv run lolpredictor train \
  --database var/current-release-v2/matches.duckdb \
  --config configs/current-release.yaml \
  --registry var/current-release-v2/evaluation-artifacts \
  --reports var/current-release-v2/release-reports

uv run lolpredictor release-gate \
  --config configs/current-release.yaml \
  --training-report var/current-release-v2/release-reports/training-report.json \
  --backtest-report var/current-release-v2/backtest-reports/backtest-report.json \
  --output var/current-release-v2/release-reports/promotion-report.json
```

Refit the gate recommendation and predict in a separate process:

```bash
uv run lolpredictor refit \
  --database var/current-release-v2/matches.duckdb \
  --config configs/current-release.yaml \
  --training-report var/current-release-v2/release-reports/training-report.json \
  --promotion-report var/current-release-v2/release-reports/promotion-report.json \
  --registry var/current-release-v2/production-artifacts \
  --reports var/current-release-v2/production-reports

uv run lolpredictor predict \
  --artifact var/current-release-v2/production-artifacts/leaguepedia-2020-2026-current-release-production-elo_only-20260730T204947Z-af54eaaa \
  --input examples/current_sample_draft.json \
  --output var/current-release-v2/production-reports/sample-prediction.json
```

Run the complete synthetic training, prediction, and screenshot workflow:

```bash
uv run lolpredictor run-all --workdir var/fixture-run
```

## Remaining risks and next milestone

Leaguepedia is community-maintained and may revise historical page names or records.
The current extraction includes lower-tier, amateur, showmatch, and other non-target competitions because tournament tier and official-circuit metadata are not yet in the snapshot contract.
That broad source mix may contribute to major-league instability.

First-pick coverage is limited, and Fearless Draft exclusions are not yet reconstructed from source coverage.
No real broadcast screenshot corpus has been supplied, so unattended screenshot parsing is not promoted.
Raw source availability depends on Fandom and should be mirrored only under compatible terms.

The next model milestone is a preregistered tier-aware or partially pooled league model.
It should add versioned tournament-level metadata, develop only on pre-2026 rolling folds, and freeze a new post-July 2026 holdout before promotion.
The immediate screenshot milestone is to label at least 100 frames from at least three recordings and ten matches for each target overlay profile using the corpus contract in `docs/overlay-corpus.md`.

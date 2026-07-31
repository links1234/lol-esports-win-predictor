# V3 chronological calibration results

## Status

The calibration follow-up is complete for development and shadow prediction.
The selected model is a 50 percent raw Elo and 50 percent native numeric CatBoost ensemble with no output mapper.
It is a development model and not a promoted production model.

The experiment followed `docs/calibration-experiment-protocol.md`.
The selection was frozen on pre-2026 rolling folds before the already-opened January through March 2026 diagnostic was run.
No April through July 2026 outcome was evaluated by this workflow.

## Data and boundaries

The experiment reused the corrected immutable source and point-in-time feature table documented in `docs/v3-model-results.md`.
The source dataset fingerprint is `f4e69cdbfd0e5d7cf0df91437d1beba797bb0d64cf664bde4a34ede3dba2ce67`.
The modeling population contains 22,162 official Primary or Premier games.

The three rolling test intervals contain 6,087 games from 2,638 series.
The first rolling test starts on September 2, 2023.
The last rolling test ends on November 9, 2025.
Every fold keeps fit, calibration, validation, and test timestamp groups strictly ordered and disjoint.

The opened diagnostic contains 744 games from January 14 through March 30, 2026.
The current snapshot's April through July 2026 interval remains unevaluated by this workflow and cannot be used as fresh promotion evidence because an earlier release cycle already opened it.

## Implemented calibration contract

The calibration layer now supports native, Platt, monotone beta, and isotonic probability mappings.
Each mapper is serializable and records its method, estimator type, sample and class counts, raw-probability range, and calibration timestamps.
The beta mapper constrains both slopes to be nonnegative so its output remains monotone.
Isotonic extrapolation is clipped to the calibration range.

The new ensemble calibration candidates fit raw Elo and raw CatBoost components only on the estimator-fit interval.
Their optional final mapper sees only the later trailing calibration probabilities and labels.
Changing calibration labels cannot refit either component.

Native final refits now use every available training row instead of discarding an unused calibration block.
Mapped final refits still reserve a later chronological calibration interval.

## Rolling comparison

Log loss is the primary metric.
The table is limited to the matched calibration families and the two principal controls.

| Candidate | Log loss | Brier | ECE | ROC-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50% raw Elo + 50% native CatBoost, native output | **0.610845** | 0.211731 | 0.022764 | 0.723759 | 66.19% |
| Native numeric CatBoost | 0.610922 | **0.211728** | 0.016061 | 0.723241 | **66.32%** |
| 50% raw Elo + 50% native CatBoost, beta | 0.611078 | 0.211837 | 0.015760 | 0.722802 | 66.22% |
| 50% raw Elo + 50% native CatBoost, Platt | 0.611227 | 0.211890 | 0.018760 | 0.722947 | 66.08% |
| Beta-calibrated numeric CatBoost | 0.611584 | 0.211992 | 0.016706 | 0.722302 | 66.12% |
| Platt-calibrated numeric CatBoost | 0.611619 | 0.212009 | 0.019532 | 0.722463 | 66.09% |
| Team-roster logistic | 0.612719 | 0.212665 | **0.009380** | 0.719184 | 65.65% |
| Isotonic-calibrated numeric CatBoost | 0.614270 | 0.213164 | 0.013376 | 0.717989 | 66.01% |
| 50% raw Elo + 50% native CatBoost, isotonic | 0.617496 | 0.212722 | 0.023423 | 0.719371 | 65.83% |
| Elo only | 0.619977 | 0.215634 | 0.021001 | 0.712263 | 65.85% |

Every row contains the same 6,087 rolling test games.
Beta reduced ensemble ECE from 0.022764 to 0.015760 but increased log loss by 0.000232.
Platt increased log loss by 0.000382.
Isotonic overfit the trailing intervals and failed the fold-stability, simple-control, paired-uncertainty, and major-league gates.

The native output therefore remained selected under the frozen primary-metric rule.
Its log loss is 0.610845 versus 0.619977 for Elo.
Its series-clustered 95 percent log-loss interval is 0.602086 to 0.620341.
Its paired candidate-minus-Elo interval is -0.011869 to -0.006209.

The best simple control remains team-roster logistic at 0.612719.
The selected-minus-simple-control interval is -0.005223 to 0.001542.
That interval crosses zero, so the evidence still does not establish superiority over the simpler logistic control.

The selected model improves Elo by 0.014624, 0.010655, and 0.002088 log loss in the three folds.
It passes every fixed development eligibility check.
Its LCS regression is 0.009831 on 192 games, only 0.000169 inside the fixed 0.01 limit.
This narrow margin is a material robustness risk.

## Opened 2026 diagnostic

After selection was frozen, the selected model alone was refitted through November 9, 2025 and evaluated on the opened 744-game diagnostic.
The native estimator used all 20,380 pre-diagnostic training games.
Elo used 18,356 estimator-fit games and a later 2,024-game Platt calibration interval.

The first diagnostic run exposed that native refits were incorrectly discarding the unused trailing 2,024-game calibration block.
That pre-fix run scored 0.666372 log loss and 0.058468 ECE, and its report is preserved locally as `confirmation-report-before-native-refit-fix.json`.
The refit policy was corrected from the data-use contract rather than chosen from diagnostic performance.
The authoritative all-data diagnostic below is slightly worse in log loss and better in ECE, which is why neither run is used to tune or promote the model.

The selected model scored 0.668256 log loss, 0.237751 Brier score, 0.054683 ECE, 0.636103 ROC-AUC, and 59.14 percent accuracy.
Elo scored 0.695109 log loss, 0.248127 Brier score, 0.072408 ECE, 0.612155 ROC-AUC, and 57.80 percent accuracy.
The paired selected-minus-Elo interval is -0.036375 to -0.017555 across 316 series.

The selected model still fails the fixed 0.04 ECE target.
This diagnostic supports a useful ranking signal but confirms that temporal probability drift remains unsolved.
It is not promotion evidence.

## Shadow artifact

The resulting artifact is:

```text
var/v3-calibration/artifacts/leaguepedia-2020-2026-v3-calibration-development-elo_catboost_numeric_raw_blend_50-20260731T023254Z-8c204599
```

Its model version is `v3-calibration-20260730+cfg.8c204599.data.f4e69cdb`.
Its feature-state cutoff is July 30, 2026 at 19:52 UTC.
Both estimator components use all 22,162 eligible games.
The manifest explicitly records native output mapping, equal component weights, zero calibration samples, and `artifact_purpose: development`.
Every prediction retains the non-promoted development warning.

A separate prediction process loaded the artifact and evaluated `examples/lck_hypothetical_draft.json`.
It returned 70.7538 percent blue and 29.2462 percent red.
That number is a model estimate for a hypothetical saved draft, not a verified statement about a played game's true pregame odds.

## Verification

The complete suite passes 93 tests.
Statement coverage is 89.62 percent against an 85 percent CI floor.
Ruff formatting and linting, strict mypy checking, lock validation, repository hygiene, credential-pattern scanning, source and wheel builds, artifact checksums, and separate-process prediction all pass.
The backtest report SHA-256 is `c2c1f5427a24e3a37f0d262baa6741ab3177ee6c7fc0d06f395d7099e1efd250`.
The authoritative confirmation report SHA-256 is `4dd4005cb3905ce1c86ecfe5cfe841df4f35c22f023ff7fde51c1fa0af4e4bf5`.

## Exact commands

Install the locked environment:

```bash
uv sync --locked --all-groups
```

Run the pre-2026 rolling comparison:

```bash
uv run lolpredictor backtest \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-calibration-development.yaml \
  --reports var/v3-calibration/reports
```

Evaluate the frozen candidate on the already-opened diagnostic:

```bash
uv run lolpredictor confirm-selection \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-calibration-development.yaml \
  --backtest-report var/v3-calibration/reports/backtest-report.json \
  --output var/v3-calibration/reports/confirmation-report.json
```

Build the warned shadow artifact:

```bash
uv run lolpredictor refit-development \
  --database var/v3-development/matches.duckdb \
  --config configs/v3-calibration-development.yaml \
  --backtest-report var/v3-calibration/reports/backtest-report.json \
  --confirmation-report var/v3-calibration/reports/confirmation-report.json \
  --registry var/v3-calibration/artifacts \
  --reports var/v3-calibration/reports
```

Run a structured prediction:

```bash
uv run lolpredictor predict \
  --artifact var/v3-calibration/artifacts/leaguepedia-2020-2026-v3-calibration-development-elo_catboost_numeric_raw_blend_50-20260731T023254Z-8c204599 \
  --input examples/lck_hypothetical_draft.json
```

Run all quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/lolpredictor
uv run pytest --cov=src/lolpredictor --cov-report=term-missing --cov-fail-under=85
uv build
```

## Decision and next milestone

The calibration methods are retained as explicit candidates, but none replaces native output under the primary log-loss rule.
The new shadow artifact is modestly better than the prior v3 selection in rolling log loss and remains materially better than Elo.
It does not meet the opened 2026 calibration target and is not safe to present as a highly accurate production probability.

The next model milestone remains a preregistered tier-aware or partially pooled league model.
It should address isolated regional rating pools and the near-threshold LCS behavior without hard-coded team identities.
Its feature family, calibration policy, and gates must be frozen on pre-2026 development data.
Promotion requires at least 750 new post-July 2026 games from 300 series that have not been opened during development.

The next screenshot milestone remains a rights-cleared labeled corpus with at least 100 target-overlay frames from at least three recordings and ten matches.
Screenshot extraction and model probability quality must continue to have separate release gates.

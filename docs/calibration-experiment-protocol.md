# Chronological calibration experiment protocol

## Status

This protocol was fixed on July 30, 2026 before running the newly implemented beta, isotonic, or blend-level calibration candidates.
It governs model development only and cannot produce a promoted production model.

## Objective

The experiment tests whether a probability mapper can reduce temporal calibration drift without sacrificing future log loss or league robustness.
Log loss remains the primary metric.
Expected calibration error, Brier score, ROC-AUC, accuracy, sample counts, and all configured breakdowns remain secondary diagnostics.

## Fixed data boundaries

The input is the corrected immutable Leaguepedia snapshot documented in `docs/v3-model-results.md`.
The modeling population remains official Primary and Premier competition only.
Each rolling fold has an expanding estimator-fit interval, a later trailing calibration interval, a later validation interval, and a later test interval.
No timestamp group may cross an interval boundary.

The rolling development data ends before January 1, 2026.
January through March 2026 is an already-opened diagnostic and may be inspected only after the development selection is frozen.
April through July 2026 has already been opened by an earlier release cycle and will not be reused for model selection or promotion.
No post-July 30, 2026 result may be opened by this experiment.

## Frozen calibration candidates

The native numeric CatBoost, Platt-calibrated numeric CatBoost, beta-calibrated numeric CatBoost, and isotonic-calibrated numeric CatBoost form one matched calibration comparison.
All four use an identical estimator family, feature contract, fit interval, and hyperparameters.

The 50 percent raw Elo and 50 percent native numeric CatBoost ensemble is compared with native, Platt, beta, and isotonic output mappings.
Both raw component estimators are fitted only on the estimator-fit interval.
The optional final mapper is fitted only on their combined probabilities in the trailing calibration interval.
Calibration labels never update either component estimator.

The previously evaluated candidates remain in the comparison as controls.
No neural model is eligible in this experiment.

## Mapper definitions

Native probabilities receive no post-estimator mapping.
Platt calibration fits a regularized logistic mapping to the raw log odds.
Beta calibration fits a monotone mapping of log probability and negative log complement probability with nonnegative slopes.
Isotonic calibration fits a nondecreasing piecewise-constant mapping and clips predictions outside the observed calibration range.
Every emitted probability is bounded away from zero and one before log-loss evaluation.

If a calibration interval contains only one outcome class or constant raw predictions, the mapper abstains and the candidate emits native probabilities.
The artifact must record that no mapping was applied in that case.

## Frozen selection rule

The existing development eligibility rule remains unchanged.
An eligible candidate must beat Elo and the best simple control on aggregate rolling log loss, have a series-clustered paired interval below zero against Elo, improve or match Elo in every fold, keep expected calibration error at or below 0.04, and keep every sufficiently large configured major-league regression within 0.01 log loss.

Among eligible candidates, the candidate with the lowest aggregate rolling test log loss is selected.
If no candidate is eligible, Elo remains the fallback.
Accuracy cannot override any failed log-loss, uncertainty, calibration, or league-robustness check.

After selection is frozen, the selected candidate alone may be evaluated on the opened January through March 2026 diagnostic.
That diagnostic may reject a shadow artifact but cannot establish promotion.

## Artifact contract

Any resulting artifact must remain a development artifact.
Its manifest must identify the calibration method, mapper estimator, calibration sample count, class counts, raw-probability range, and calibration start and end timestamps.
It must also describe component weights and component-level calibration.
Predictions must retain the development-model warning.

## Acceptance criteria

- Unit tests prove that all mappings are finite, bounded, monotone, and serializable.
- Leakage tests prove that changing calibration labels cannot refit the ensemble components.
- Calibration provenance ends before validation begins in every tested split.
- The full synthetic workflow trains, saves, reloads, and predicts with the expanded candidate set.
- Rolling comparisons use only pre-2026 development outcomes.
- The selected method is frozen before the opened diagnostic is run.
- No April through July 2026 or post-July 2026 evaluation is performed.

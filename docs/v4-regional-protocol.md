# V4 partially pooled regional-strength protocol

## Status

This protocol was fixed before generating regional outcome features or running the new regional candidates.
It defines a development and shadow experiment only.
It cannot promote a model because every outcome in the current snapshot has already been opened by an earlier development or release cycle.

## Objective

The experiment tests whether sparse international results can improve future pregame probabilities by propagating regional strength to teams that mostly play within isolated domestic rating pools.
It does not assume that region must help.
Log loss remains the primary metric.

## Label-blind metadata audit

The official Primary or Premier target population contains 22,162 games, 22 league labels, 15 region labels, and no missing region or tournament-level values.
Every eligible game is currently labeled `Primary`.
There is no `Premier` example and no within-population tournament-tier variation.

The current source therefore cannot identify a tier effect without expanding the target into Secondary events.
This experiment will not change the target population or treat Secondary games as equivalent to top professional competition.
No tier candidate will be created from a constant field.

The source contains 1,245 official Primary or Premier games labeled as international competition.
Earlier domestic-event metadata supplies usable home-region history for most international participants after the beginning of the corpus.
The source taxonomy changes over time, including Europe to EMEA and several later Americas and Asia Pacific consolidations.
These labels remain separate time-varying rating pools because no versioned external mapping has been validated.

## Home-region contract

A team's home region for a domestic-event request is the request's known non-excluded event region.
A team's home region for an excluded international-event request is its most recently observed non-excluded event region from an earlier timestamp group.
A current match may use its own pregame event metadata, but its outcome cannot update any state until every match at that timestamp has been featurized.

The configured excluded event-region labels contain only `International`.
No team name or team ID is mapped to a region in code.
Unknown home regions remain unknown and produce neutral regional features.

After a timestamp group is complete, non-excluded event metadata updates each participating team's stored home region.
Source taxonomy changes therefore take effect prospectively without rewriting older feature rows.

## Regional meta-Elo contract

Every regional rating starts at 1500.
The regional K factor is fixed at 12.
A result updates regional ratings only when all of the following pregame structural conditions hold:

- The match is explicitly official.
- The tournament level is `Primary` or `Premier`.
- Both teams have known home regions.
- The home regions differ.

Same-region games do not update regional ratings.
Unofficial, Secondary, Showmatch, or missing-level games do not update regional ratings.
All games at one timestamp use the same pre-update ratings, and their deltas are accumulated before application.

## Frozen regional features

The following numeric features are added to the point-in-time contract:

- Blue and red regional meta-Elo.
- Regional meta-Elo difference.
- Regional Elo-implied blue win probability.
- Blue and red log cross-region evidence counts.
- Cross-region evidence-count difference.
- Blue and red known-home-region indicators.
- Cross-region-match indicator.
- Team-plus-region pooled Elo difference.
- Team-plus-region pooled Elo-implied blue win probability.

The pooled rating difference is the existing inactivity-adjusted team Elo difference plus the regional meta-Elo difference.
Models may learn to ignore either component.

A dedicated regional provenance timestamp records only earlier cross-region results used by the two relevant region ratings.
It must be strictly earlier than the target match timestamp.

## Frozen candidates

The existing v3 candidates remain unchanged and use the pre-regional numeric feature contract as matched controls.
The new candidates are:

- Regional Elo logistic using team Elo, regional rating, coverage, and cross-region indicators.
- Team-roster regional logistic using the prior team-roster contract plus all regional features.
- Full regional logistic using the prior full numeric contract plus all regional features.
- Full regional histogram gradient-boosted trees.
- Native full regional CatBoost.
- A fixed 50 percent raw Elo and 50 percent native regional CatBoost ensemble.

The CatBoost and ensemble candidates retain native output mapping because the frozen calibration follow-up did not improve primary log loss.
The logistic and histogram-tree candidates retain the existing trailing Platt mapping.
No weight search, regional K-factor search, neural model, identity embedding, or post-2025 tuning is allowed.

## Temporal evaluation

The experiment reuses the three expanding rolling-origin folds from the v3 calibration run.
Estimator fitting, probability calibration, validation, and rolling tests remain strictly chronological and timestamp-grouped.
The combined rolling test interval ends on November 9, 2025.

The existing eligibility gates remain unchanged.
An eligible candidate must beat Elo and the best simple control, have a series-clustered paired interval below zero against Elo, improve or match Elo in every fold, keep ECE at or below 0.04, and stay within the fixed 0.01 major-league regression limit.

Among eligible candidates, aggregate rolling test log loss selects the development model.
If no changed candidate improves the existing selection under that rule, the existing native Elo-CatBoost blend remains the shadow recommendation.

After the selection is frozen, the selected candidate alone may be evaluated on the already-opened January through March 2026 diagnostic.
April through July 2026 will not be evaluated by this workflow.
No post-July 2026 outcome may be opened.

## Acceptance criteria

- Region metadata and tier limitations are reported without changing the target population.
- Region state is serializable and prediction-equivalent after round-trip loading.
- Simultaneous matches cannot update each other's regional features.
- Flipping a target or future outcome cannot change that match's regional features.
- Ineligible match levels cannot change regional ratings.
- Unknown home regions emit finite neutral values.
- Historical and live feature rows remain identical.
- Existing candidates retain their exact pre-regional feature contracts.
- The synthetic fixture exercises domestic, cross-region, unknown-region, serialization, training, artifact loading, and prediction paths.
- The full CI workflow passes with at least 85 percent statement coverage.

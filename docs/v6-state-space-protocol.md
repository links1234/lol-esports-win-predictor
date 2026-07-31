# V6 hierarchical state-space protocol

## Status

This protocol is frozen before any v6 challenger is evaluated on real outcomes.
It defines a development experiment and cannot promote a production model.
The fixed v4 regional Elo-CatBoost model remains the control and the current shadow recommendation.

All outcomes from 2026 are already open from earlier development work.
No 2026 outcome may influence a v6 state parameter, fitted coefficient, probability mapper, candidate choice, stopping decision, or result interpretation.

## Research basis

Glicko extends Elo by representing strength with both a mean and an uncertainty and by increasing uncertainty during inactivity.
Its rating-period update also computes every participant's change from the same pre-period state.
The primary descriptions are Mark Glickman's [Glicko system paper](https://www.glicko.net/glicko/glicko.pdf) and [research page](https://www.glicko.net/research.html).

TrueSkill generalizes uncertainty-aware rating to teams and infers individual contributions from team results through approximate Gaussian inference.
TrueSkill Through Time adds Gaussian skill drift between time steps.
The primary references are the [TrueSkill technical report](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system-2/) and the [TrueSkill Through Time paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf).

V6 adopts the useful filtering concepts without claiming to implement Glicko or TrueSkill exactly.
It uses an auditable diagonal Gaussian approximation designed for the project's binary, team-based, post-draft prediction contract.

## Objective

The objective is to determine whether explicit uncertainty, roster transitions, and additive partially pooled strength components improve calibrated post-draft pregame probabilities over v4.
Log loss remains the primary selection and comparison metric.
Brier score, expected calibration error, ROC-AUC, accuracy, sample count, time, patch, league, region, and tournament breakdowns remain diagnostics.

This experiment tests one modeling hypothesis through a small fixed candidate set.
It is not another general hyperparameter search.
No neural architecture is in scope.

## Frozen data boundary

The exclusive supervised development cutoff is `2026-01-01T00:00:00Z`.
DuckDB must be queried with this cutoff before Python receives any source match.
Feature replay must start from empty state and may process only matches strictly earlier than the cutoff.

The supervised modeling population is limited to explicitly official `Primary` or `Premier` games.
The state-space filter is updated only by the same official `Primary` or `Premier` population.
Rows with identical match timestamps form one indivisible rating period and cannot observe one another's outcomes.

No January through July 2026 result is available to selection or evaluation.
A post-lock development refit may estimate the already-frozen model on later data, but it creates no evaluation evidence and must remain visibly marked as development-only.

## Latent model

Each latent component has a Gaussian approximation with a mean, variance, last-observed timestamp, and prior game count.
The pregame linear predictor contains additive signed components for global blue-side advantage, league-specific blue-side advantage, region, team, player, exact roster, role-aware champion, patch-role-champion deviation, and role matchup.

Blue entities receive positive design coefficients and red entities receive negative design coefficients.
The role matchup term uses one canonical unordered champion pair and reverses its coefficient when its orientation is reversed.
This makes the matchup effect antisymmetric and prevents two unrelated directional estimates.

The component weights and centered prior variances are frozen as follows.

| Component | Design coefficient | Prior variance |
| --- | ---: | ---: |
| Global blue side | `1.0` | `0.04` |
| League blue side | `1.0` | `0.04` |
| Region | `0.5` per side | `0.36` |
| Team | `1.0` per side | `0.64` |
| Player | `0.6 / 5` per player | `0.64` |
| Exact team roster | `0.35` per side | `0.25` |
| Role champion | `0.5 / 5` per pick | `0.16` |
| Patch-role-champion deviation | `0.35 / 5` per pick | `0.09` |
| Role matchup | `0.4 / 5` per role | `0.16` |

All new components begin at mean zero.
This is partial pooling because sparse identities and interactions remain close to their parent zero-centered effect unless repeated earlier evidence moves them.
The global role-champion term persists across patches, while the patch-role-champion deviation begins again at zero on a new patch.

Means decay toward zero with a 540-day half-life.
Posterior variances return toward their component prior variance with a 180-day half-life.
Posterior variance is bounded below by `0.0001`.
These projections are computed at the target timestamp without mutating historical state.

For one draft, let the projected linear mean be `m` and its independent-component variance be `s2`.
The uncertainty-aware raw probability is frozen as:

```text
p = sigmoid(m / sqrt(1 + pi * s2 / 8))
```

This approximation moves an uncertain forecast toward 50 percent instead of treating an imprecise mean as certain.

For a complete timestamp group, every design row and probability is computed from the same projected pre-timestamp state.
For each latent component, gradients and diagonal curvature are accumulated across the whole group.
The frozen update is:

```text
q = sqrt(1 + pi * s2 / 8)
gradient_i += (x_i / q) * (y - p)
precision_i += (x_i / q)^2 * p * (1 - p)
v_i_new = 1 / (1 / v_i + precision_i)
m_i_new = m_i + v_i_new * gradient_i
```

The update is deterministic and independent of row order within a timestamp.
It is an approximate diagonal Laplace filter and intentionally does not claim an exact joint Bayesian posterior.

## Frozen feature contract

The filter emits its raw probability, latent linear mean, latent standard deviation, signed component contributions, side contribution, team and roster uncertainty, and the fraction of weighted components with earlier observations.
The same function produces these fields for historical training rows and live prediction requests.
The serialized feature state contains every latent mean, variance, observation count, and timestamp needed to reproduce a later prediction.

The following leakage rules are mandatory:

- A target outcome cannot change any feature on its own row.
- A future outcome cannot change an earlier row.
- Outcomes at the same timestamp cannot change one another's features.
- Reordering rows inside one timestamp cannot change features or final state.
- A serialized and reloaded state must produce identical probabilities and uncertainty.
- Unknown identities must receive neutral means and configured prior uncertainty.
- No latent last-observed timestamp may equal or exceed its target timestamp.

## Fixed challengers

Exactly five v6 challengers are allowed.
Their order is frozen.

1. `state_space_native` uses the raw uncertainty-integrated state-space probability without a fitted mapper.
2. `state_space_platt` fits only a Platt probability mapper on the trailing 10 percent of each history interval.
3. `state_space_augmented_logistic` fits L2 logistic regression with `C=0.10` on core, team-strength, roster-form, draft-interaction, regional, and state-space numeric groups.
4. `state_space_v4_blend_15` assigns 85 percent weight to the fixed v4 probability and 15 percent to the augmented logistic probability.
5. `state_space_v4_blend_30` assigns 70 percent weight to the fixed v4 probability and 30 percent to the augmented logistic probability.

The augmented model deliberately excludes sparse categorical identities, independent patch categorical terms, and the old sparse champion-rate group.
The dynamic filter already represents identities and draft effects with shrinkage.
The v5 study found no stable evidence that high-capacity categorical trees or sparse rate features improved the fixed v4 control.

The v4 control retains its exact pre-v6 numeric feature contract.
Adding new fields to the latest feature table must not silently add them to v4.

## Nested chronological evaluation

Four expanding-window outer folds estimate the complete v6 selection policy.
Each outer history contains three expanding-window inner folds.
Timestamp groups remain indivisible in every partition.

Each challenger is fitted and scored on every inner fold.
The challenger with the lowest pooled inner log loss is eligible only if all of these conditions hold:

- Pooled inner log loss is lower than fixed v4.
- Pooled inner log loss is lower than Elo-only.
- Pooled inner expected calibration error is at most `0.04`.
- No inner fold regresses more than `0.005` log loss against v4.
- No configured major league with at least 100 inner examples regresses more than `0.01` log loss against v4.

If no challenger is eligible, fixed v4 wins that outer fold.
The selected candidate is refitted on the complete outer history and touches the outer score interval once.
Outer outcomes never alter a candidate, parameter, mapper, eligibility rule, or feature choice.

The final locked specification is the candidate chosen from the fourth outer fold's inner evidence.
The fourth outer result may characterize that locked choice but may not change it.

## Decision gates

V6 replaces v4 as the shadow recommendation only if the pooled outer selection policy satisfies every gate below.

- It improves v4 log loss by at least `0.003`.
- Its series-clustered paired 95 percent interval against v4 has an upper bound below zero.
- Its expected calibration error is at most `0.04`.
- It has at least 750 games and 300 series.
- No outer fold regresses more than `0.01` log loss against v4.
- No configured major league with at least 100 pooled outer examples regresses more than `0.01` log loss against v4.

Accuracy cannot override a failed log-loss gate.
A better point estimate with an interval crossing zero is reported as uncertain.
A failed gate leaves v4 unchanged.

No candidate can be called production-ready because there is no untouched post-July 2026 promotion interval.

## Reproducibility

The study writes a resolved configuration, bounded dataset fingerprint, replay summary, inner results, outer predictions, paired uncertainty intervals, gate decision, and locked finalist to an ignored study directory.
Every report uses JSON with finite numbers and stable key ordering.
The locked finalist includes a canonical SHA-256 fingerprint of its full specification.

A separate refit command must validate the lock before creating a development artifact.
The artifact must include the v6 feature state, complete candidate specification, feature and data cutoffs, dependency versions, checksums, and development-only warning.

## Acceptance criteria

- Configuration validation rejects any unregistered candidate, reordered candidate list, nonexclusive cutoff, or altered frozen parameter.
- Unit tests cover time projection, uncertainty contraction after evidence, inactivity expansion, order-independent batch updates, unknown entities, and numerical bounds.
- Leakage tests mutate target, same-timestamp, and future outcomes.
- Serialization tests prove exact prediction equivalence.
- Integration tests prove SQL cutoff enforcement and fixed v4 feature isolation.
- A synthetic end-to-end test replays data, performs nested selection, writes and validates a lock, refits an artifact, loads it in a separate process, and predicts a later draft.
- The complete real pre-2026 comparison either passes every decision gate or explicitly retains v4.
- Ruff formatting and linting, strict mypy, lock validation, package builds, repository hygiene, credential scanning, and the full test suite pass.

## Interpretation boundary

The state-space model estimates associations in historical professional match outcomes.
It cannot prove draft causality, observe private scrim information, know unannounced roster conditions, or guarantee a game result.
Its uncertainty is model uncertainty under a diagonal approximation, not a complete measure of every real-world unknown.
Screenshot extraction quality remains a separate system with separate precision, exact-draft, and abstention measurements.

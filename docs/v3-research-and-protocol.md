# V3 research and development protocol

## Decision

V3 will improve the post-draft pregame probability model without relaxing any point-in-time boundary.
The model-selection target remains log loss, not accuracy.
The implemented milestone prioritizes source correctness, target-population validity, stable role-level skill features, and calibrated tree models.
Raw identity models will remain comparison candidates rather than assumed improvements.
A neural model remains deferred until an auditable non-neural candidate has been exhausted and a neural candidate wins the same temporal backtest.

## Prediction target

The target is the probability that the blue-side team wins a professional League of Legends game after the completed draft is known and before gameplay begins.
The input state includes map side and first-pick order separately.
This distinction is mandatory in 2026 because Riot's First Selection system allows one team to choose side while the opponent chooses whether to draft first or second.
The target also includes the known players, role-aligned champions, patch, competition, series score, bans when available, and prior series exclusions when known.
No live gameplay event, postgame statistic, current-game betting movement, or current-game result is eligible.

## Competitive factors supported by primary evidence

### Side and draft order

Riot introduced First Selection for the 2026 season because side win rates varied and because side and first-pick preference represent different strategic choices.
The official description says that the team with First Selection chooses either map side or first versus second pick, while the opponent chooses the remaining property.
Source: [Riot Games, Season Start 2026](https://lolesports.com/en-AU/news/season-start-2026-lol-esports).

V3 therefore models blue side, first-pick side, and missing first-pick information separately.
It never derives first pick from blue side for 2026 games.

### Fearless Draft and series state

Riot expanded Fearless Draft across the 2025 professional season.
Previously selected champions can become unavailable in later games, so prior games in the same series are part of the pregame draft state.
Sources: [Riot Games, LoL Esports in 2025](https://lolesports.com/en-GB/news/lol-esports-in-2025) and [Riot Games, Fearless Draft Takes Over 2025](https://lolesports.com/en-PH/news/fearless-draft-takes-over-2025).

V3 treats game number, prior series score, prior series picks, and known Fearless exclusions as temporal features.
The system will not infer that every lower-tier competition used the same Fearless rules.

### Player, champion, and player-champion skill

Player Skill Decomposition evaluates player base skill, champion base skill, and player-champion-specific skill as distinct components.
Its League of Legends results support retaining all three rather than collapsing a draft into team-average champion win rate.
Source: [Chen et al., Player Skill Decomposition in Multiplayer Online Battle Arenas](https://arxiv.org/abs/1702.06253).

The player-champion experience study also found useful signal in champion-specific player experience.
However, it used a random 80/20 split and stratified cross-validation on ranked games, so its reported 75 percent accuracy is not a valid estimate for future professional matches.
It also found gradient boosting marginally ahead of its deep neural network, which does not support a neural-first implementation.
Source: [Do et al., Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience](https://arxiv.org/abs/2108.02799).

V3 adds role-level player Elo, inactivity-adjusted ratings, role-level champion history, player-champion games and win rates, matchup history, and coverage counts.

### Draft interactions and recent histories

DraftRec models player preference, team synergy, opponent competence, and player history with a hierarchical architecture.
It used a chronological 85/5/10 split and reported that history-aware models performed better than no-history variants.
It also found prominent top-jungle, mid-jungle, and bottom-support interactions.
Source: [Lee et al., DraftRec](https://arxiv.org/abs/2204.12750).

DraftRec studied high-ranked ladder games rather than professional competition and reported accuracy and mean absolute error instead of calibrated log loss.
Its architecture is therefore evidence for feature families, not evidence that its published model should be copied or that its reported numbers transfer to this product.

V3 expands stable interaction summaries first.
Direct identity and high-dimensional interaction models are included only as measured ablations.

### Regional rating pools

PandaSkill identifies isolated regional rating pools as a problem and combines contextual player ratings with a regional meta-rating.
It evaluates professional matches with a rolling one-year training window and subsequent one-month tests.
Source: [De Bois et al., PandaSkill](https://arxiv.org/abs/2501.10049).

Its individual performance score uses postgame statistics.
Those statistics could update ratings for later matches, but they cannot be used for the target match and are not present in the current approved pregame source.
V3 will initially use outcome-based pregame ratings and explicit region metadata.
A later milestone may add prior-match-only performance ratings from a separately validated public source.

### Probability quality

Strictly proper scoring rules such as log loss and Brier score evaluate probabilistic predictions.
Probability calibration must be learned on data independent of the estimator's fit data.
Source: [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

V3 retains a trailing chronological calibration partition for every fit.
Calibration data is not used to fit preprocessing or the base estimator.

## Source audit findings

The corrected source contains 95,588 rows and 93,041 accepted games from 302 league labels between January 2020 and July 30, 2026.
It mixes primary leagues, secondary leagues, academy events, amateur events, and showmatches.
That mixture is useful for historical state and player movement but is not the correct unweighted target population for a primary professional prediction product.

Leaguepedia exposes competition region, tournament level, and official-circuit status in its Cargo schemas.
Sources: [Leaguepedia Tournaments schema](https://lol.fandom.com/wiki/Module%3ACargoDeclare/Tournaments) and [Leaguepedia Leagues schema](https://lol.fandom.com/wiki/Module%3ACargoDeclare/Leagues).

V3 will preserve all accepted matches for state replay while allowing the supervised fit and evaluation population to require official Primary or Premier competition.
Secondary and unknown-level games will be reported as separate diagnostics.

The audit also reproduced a source-contract defect.
Leaguepedia declares patch as a string, but CargoExport serializes numeric-looking patch strings as JSON numbers.
This collapses `25.10` into `25.1`, `25.20` into `25.2`, and equivalent patches in other years.
The v3 fetch contract forces patch to a string with a sentinel and removes that sentinel during validation.
The new immutable snapshot and schema-v3 database are now the source of the final v3 comparisons.

The official Primary target population has standard-ban coverage above 99 percent.
First-pick coverage is absent before 2026 and about 28 percent in 2026.
Fearless exclusions are not directly populated.
Missingness is therefore modeled explicitly and is never replaced with knowledge inferred from the outcome.

## Opened and prospective data boundaries

Every game through July 30, 2026 had already been inspected during v2 development.
The January through March 2026 validation interval and April through July 2026 holdout are opened diagnostics.
They cannot provide an unbiased promotion claim for v3.

V3 candidate design and hyperparameters will use only rolling-origin development tests whose target games end no later than December 31, 2025.
The folds use expanding fit intervals, trailing calibration intervals, subsequent validation intervals, and subsequent test intervals.
Complete timestamp groups stay together.
Series are the bootstrap cluster.

The next untouched promotion holdout begins after July 30, 2026.
It remains closed until the candidate family and hyperparameters are frozen and at least 750 games from at least 300 series are available.
No v3 model will be called production-better before that gate is passed.

## Frozen candidate families

The v2 candidates remain controls:

- Blue-side base rate.
- Elo-only logistic calibration.
- Team and roster logistic regression.
- Draft-only logistic regression.
- Combined logistic regression.
- Histogram gradient-boosted trees.
- Fixed Elo and logistic blend.

V3 adds the following development candidates:

- Matched legacy-feature controls for the combined logistic and histogram-tree models.
- Role-aware combined logistic regression using the expanded point-in-time numeric feature contract.
- Role-aware histogram gradient-boosted trees using the same expanded contract.
- Recency-weighted logistic regression with weights computed only from each fold's fit timestamps.
- Team-roster, legacy-feature, and complete numeric CatBoost variants.
- Native CatBoost probabilities and trailing Platt-calibrated CatBoost probabilities as explicit calibration ablations.
- Numeric CatBoost with 250 trees, depth 6, learning rate 0.04, L2 leaf regularization 8, fixed seed, and no validation-driven early stopping.
- Context CatBoost with the same numeric state plus role-aligned team, player, champion, player-champion, league, region, level, and patch categories.
- Fixed 25 and 50 percent Elo shrinkage blends with native numeric CatBoost.

CatBoost receives chronologically sorted rows and uses time order for categorical statistics.
Its official documentation states that `has_time` preserves input order instead of performing random permutations during categorical transformation and tree construction.
Source: [CatBoost parameter tuning documentation](https://catboost.ai/docs/en/concepts/parameter-tuning).

The final development comparison found that native numeric CatBoost beat the existing histogram booster, while raw identity context and recency weighting were worse.
The 25 percent Elo blend had the lowest raw log loss but failed the major-league robustness rule.
The 50 percent Elo blend passed every development rule and became the locked shadow candidate.
Full measurements are in [`v3-model-results.md`](v3-model-results.md).

## Leakage controls

Every derived historical row must satisfy all of the following:

- Team and player ratings are read before the target outcome update.
- Team form uses only earlier timestamp groups.
- Champion, patch-champion, player-champion, synergy, and matchup statistics use only earlier timestamp groups.
- Earlier games at the exact same timestamp cannot affect one another.
- Tournament metadata is structural pregame metadata and contains no result-derived field.
- Model population filtering does not depend on winner or any postgame measurement.
- Numeric preprocessing and categorical vocabularies are fitted on the fold fit partition only.
- Recency weights use the maximum timestamp inside the fit partition and never an evaluation timestamp.
- The calibrator is fitted only on the trailing calibration partition.
- Validation selects a candidate only inside a development fold.
- Test targets are not used for feature updates, refitting, calibration, hyperparameter choice, or candidate choice.
- The final prospective holdout remains unopened until the release gate is eligible.

Automated tests perturb future labels, reorder same-timestamp matches, inject extreme calibration values, inject unseen categories, and verify recorded provenance timestamps.
The tests must fail if a source timestamp is equal to or later than its target.

## Selection and acceptance criteria

Aggregate rolling log loss is the primary ranking metric.
The report also includes Brier score, expected calibration error, ROC-AUC, accuracy, sample count, calibration bins, and breakdowns by month, patch, league, region, and tournament level.
Paired log-loss differences against Elo are bootstrapped by complete series.

A candidate is eligible for the prospective holdout only if it meets all of the following on development data:

- It improves aggregate log loss over Elo and the best simpler control.
- The improvement is stable across folds rather than coming from one interval.
- It does not materially regress any major league with an adequate sample.
- Expected calibration error remains at or below 0.04.
- Unknown entities and sparse histories produce bounded probabilities and explicit warnings.
- Side swapping produces a consistent probability complement within a documented tolerance for symmetric model families.
- Training and separate-process prediction use the same feature contract.

The prospective promotion gate requires at least 0.003 lower log loss than Elo, a series-clustered paired interval that supports the improvement, the configured calibration limit, and no major-league regression above 0.01.

## Deferred work

Prior-match-only in-game performance ratings are deferred until a licensed and validated public source is added.
Champion kit embeddings and patch-note embeddings are deferred until the stable statistical features are exhausted.
A Siamese or Transformer model is deferred until the new tree and linear candidates have a reproducible ceiling.
Betting odds are excluded from the product model because they change the target from an independent draft model into a market imitation model.

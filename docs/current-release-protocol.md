# Current-data release protocol

## Status

This protocol was fixed on July 30, 2026 before inspecting any 2025 or 2026 match outcomes.
It defines the next model-selection and release decision.
Changing these rules after opening the final holdout invalidates that holdout for a release claim.

## Source scope

The preferred inputs are the public annual Oracle's Elixir files linked from the official download page.
The source file identifier, filename, byte size, modification time, retrieval time, SHA-256 checksum, schema version, row count, game count, accepted count, and quarantine count must be recorded.
The system may combine the already fingerprinted 2020-2024 research corpus with direct 2025 and 2026 files.
No post-game source field other than the result label may enter a feature.

The model will include all validated professional leagues in the source.
Results must also be reported separately for LCK, LPL, LEC, the current Americas league labels, the current Pacific league labels, First Stand, MSI, and Worlds when their sample counts permit.
League aliases may be normalized only through versioned data mappings with tests.

## Source decision recorded before the final holdout

The direct 2025 and 2026 files linked by the [Oracle's Elixir download page](https://oracleselixir.com/tools/downloads) returned Google Drive quota responses during this release run.
The release corpus will therefore use one pinned Leaguepedia Cargo extraction from 2020 through the July 30, 2026 retrieval cutoff instead of stitching two incompatible identity systems together.
This decision was recorded before evaluating any April 2026 or later outcomes.

Leaguepedia's [ScoreboardGames declaration](https://lol.fandom.com/wiki/Module%3ACargoDeclare/ScoreboardGames) identifies `GameId` as the game-level join key and states that the role-ordered team pick arrays may be used when all ten champions are required.
Leaguepedia's scoreboard implementation stores the role-ordered player link arrays, so those canonical page names are used as the player identity source.
The [Tournaments declaration](https://lol.fandom.com/wiki/Module%3ACargoDeclare/Tournaments) identifies `OverviewPage` as the tournament join key and `League` as the canonical league relationship.

The fixed extraction joins `ScoreboardGames`, `MatchScheduleGame`, and `Tournaments`.
It reads only stable IDs, timestamps, league and tournament labels, patch, teams, role-ordered rosters, picks, bans, side, selection metadata, game number, and the result label.
The snapshot stores the exact query contract, interval, retrieval timestamp, source endpoint, row count, and SHA-256 checksum.
Leaguepedia content is available under CC BY-SA 3.0 unless otherwise noted, so source attribution and the license note must remain with every ingestion run.

A label-blind count found 95,579 joined games from January 3, 2020 through July 30, 2026.
The 2026 joined population contained 8,340 rows before canonical validation.
Of those rows, 131 lacked patch, 18 lacked side assignment, and 2,334 supplied `FirstPick`.
Incomplete required records will be quarantined, and missing first-pick values will remain unknown.

## Fixed chronological boundaries

The estimator-development interval ends on December 31, 2025 at 23:59:59 UTC.
The model-selection validation interval begins on January 1, 2026 and ends on March 31, 2026 at 23:59:59 UTC.
The final release holdout begins on April 1, 2026 and ends at the latest complete source game strictly before the source retrieval timestamp.

The estimator-development interval uses an internal chronological fit and calibration split.
The last 10 percent of timestamp groups in that interval are reserved for probability calibration.
All preprocessing and estimator parameters are fitted before that calibration interval.

Rolling-origin development folds may use only games before January 1, 2026.
No rolling development fold may inspect the validation or final holdout intervals.
The final holdout is opened exactly once after source validation, leakage tests, candidate definitions, and promotion thresholds pass.

If the final holdout contains fewer than 750 accepted games or fewer than 300 series clusters, it is reported as provisional and cannot independently support a release claim.

## Fixed candidates

The candidate families are:

1. Blue-side base rate.
2. Elo only.
3. Team-and-roster logistic regression.
4. Draft-only logistic regression.
5. Combined logistic regression.
6. Gradient-boosted trees.
7. A fixed blend containing 30 percent Elo and 70 percent combined logistic probability.

The fixed blend weight is not retuned on 2026 data.
A future learned blend requires a new development protocol and a new untouched holdout.
No neural or Siamese model is eligible in this release cycle.

## Selection and promotion

Candidate selection uses validation log loss only.
The selected family is then refitted using all information available before April 1, 2026, with a new trailing chronological calibration interval.
The refitted model is evaluated once on the final holdout.

Promotion requires all of the following:

- The selected model beats blue-side base rate and Elo on validation log loss.
- The selected model beats Elo by at least 0.003 mean log loss across rolling development predictions.
- The selected model beats Elo by at least 0.003 log loss on the final holdout.
- A series-clustered paired 95 percent bootstrap interval for selected-minus-Elo holdout log loss has an upper bound below zero.
- Holdout Brier score is no worse than Elo.
- Holdout expected calibration error is at most 0.04.
- No major league with at least 100 holdout games regresses against Elo by more than 0.01 log loss.
- Probabilities are finite, complementary, and produced without unknown required champions.

Failure to satisfy a promotion condition keeps Elo or the previous simpler model as the production choice.
Accuracy cannot override a failed log-loss or calibration gate.

## Current-season data-quality gates

Every accepted 2026 game must have:

- A timezone-aware start timestamp.
- A normalized patch.
- Two teams and five role-aligned players per side.
- Five unique champion picks per side.
- Complementary outcome labels.
- A stable game ID.
- Side assignment.
- First-pick assignment when the source supplies it.

The 2026 First Selection coverage rate must be reported.
Missing first-pick data is allowed only with an explicit warning and may not be derived from map side.
The pre-2026 blue-first-pick compatibility flag is forbidden for the 2026 file.

Series score is reconstructed only from earlier accepted games in the same series.
Fearless Draft exclusions are derived only from earlier games when source coverage permits.
Games with inconsistent role, draft, result, timestamp, series, or identity data are quarantined rather than repaired silently.

## Screenshot gate

Screenshot parsing remains a separate release decision from model promotion.
The initial real-overlay corpus must store source provenance, recording group, match group, overlay profile, frame timestamp, resolution, and verified structured labels.
Raw broadcast images remain outside Git unless redistribution rights are explicit.

Automatically accepted fields require at least 99.5 percent precision on recording-grouped holdouts.
All ten champion slots require at least 95 percent exact-draft accuracy on supported overlay profiles.
Unsupported layouts and uncertain required fields must abstain.

Model metrics must never be reported as screenshot-to-outcome metrics until both parser and probability gates pass independently.

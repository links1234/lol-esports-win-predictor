# Live broadcast draft prediction plan

## Product definition

The user attaches a screenshot from a professional League of Legends broadcast after all ten champions are visible.
The system extracts the teams and role-aligned champions, combines them with confirmed pre-game context, and returns a calibrated pre-game win probability.
The broadcast may be live, but the model deliberately ignores every event that occurred after the game began.

The first release is an analysis tool.
It is not a betting, staking, or gambling product.

## Accuracy strategy

There are two independent accuracy problems.
The parser must reconstruct the structured draft correctly.
The probability model must generalize to future professional games and remain calibrated.

Parser and model metrics must never be blended into one headline number.
A correct model cannot recover from a wrong champion or team parse, and a perfect parse does not imply that the game outcome is highly predictable.

The probability objective is minimum future log loss.
Accuracy remains a secondary descriptive metric because it ignores confidence and calibration.

## Delivery phases

### Phase A - Real-data and temporal evaluation foundation

Acceptance criteria:

- A pinned Oracle's Elixir adapter reads an explicit pre-game column allowlist plus the outcome label.
- Altering any post-game source column cannot change a canonical draft record or historical feature.
- Invalid games are quarantined with machine-readable reason codes.
- Source filename, URL when supplied, retrieval time, SHA-256 checksum, schema version, and accepted and rejected counts are recorded.
- Rolling-origin folds keep complete timestamp groups together.
- Every fold fits preprocessing, estimators, and calibration only on data earlier than its validation and test intervals.
- Reports compare base rate, Elo-only, team and roster, draft-only, combined logistic regression, and gradient-boosted trees.
- Reports include log loss, Brier score, calibration, ROC-AUC, accuracy, sample counts, temporal slices, league slices, patch slices, and series-clustered uncertainty.

### Phase B - Screenshot extraction trust boundary

Acceptance criteria:

- Overlay profiles and template catalogs are versioned and checksum-addressed.
- The parser emits a candidate, confidence, and runner-up for every required field.
- Low-confidence required fields prevent prediction until corrected or confirmed.
- Screenshot context rejects all unknown fields, including in-game state.
- Pixels outside configured crop regions cannot affect the parse.
- A synthetic overlay fixture exercises parsing, confirmation, artifact loading, and prediction in separate processes in CI.

### Phase C - Supported real broadcast overlays

Acceptance criteria:

- A labeled benchmark contains multiple recordings, tournaments, resolutions, compression levels, animation frames, and overlay revisions.
- Splits are grouped by recording and match.
- Automatically accepted per-slot precision is at least 99.5 percent.
- Exact ten-champion draft accuracy is at least 95 percent for supported profiles.
- Team-name and champion corrections are editable before prediction.
- Unsupported layouts abstain instead of returning a probability.

### Phase D - Production model promotion

Acceptance criteria:

- Training spans multiple seasons with an untouched recent chronological holdout.
- A promoted model beats base rate and Elo on mean rolling-fold log loss and the final holdout.
- The improvement is stable across major leagues and is not concentrated in one patch or tournament.
- Calibration error and Brier score do not materially regress.
- Release metrics include series-clustered confidence intervals and sample counts.
- The artifact cutoff is recent enough for the target match, or the product shows a stale-data warning.

### Phase E - Live-series freshness

Acceptance criteria:

- Completed earlier games in the same series can be applied as a transient point-in-time overlay.
- Series score, roster substitutions, First Selection, and Fearless Draft exclusions are explicit inputs.
- The current game's state can never enter the overlay.
- The overlay is discarded or persisted through the normal audited ingestion path after the series.

## Promotion rules

A more complex model is promoted only when it improves validation log loss across repeated temporal folds and improves the final untouched holdout.
A neural or Siamese model is not added merely because it can represent champion combinations.
It must beat the calibrated simpler candidates by a predeclared meaningful margin and remain robust for unknown rosters, new champions, and patch changes.

No release should promise a fixed accuracy percentage before the real-data holdout is measured.
The interface should communicate probability and uncertainty, not certainty.

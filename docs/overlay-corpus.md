# Real broadcast overlay corpus

## Purpose

The screenshot parser has a separate release gate from the probability model.
A model probability must never be shown automatically when a required screenshot field is uncertain.

The repository includes deterministic synthetic image coverage for CI.
Synthetic images cannot establish real broadcast accuracy.
A real corpus must cover the target tournament overlays before screenshot parsing can be promoted.

## Storage and rights

Store raw frames, profiles, catalogs, and the manifest together under an ignored directory such as `data/raw/vision/lck-2026-v1/`.
Do not commit broadcast frames unless redistribution rights are explicit.
Every frame records its source URL, source license, and one of `cleared`, `not_cleared`, or `unknown` for redistribution rights.
The benchmark report contains hashes, labels, and metrics but does not copy image pixels.

## Required manifest fields

The manifest uses `corpus_schema_version` 1 and contains a non-empty `frames` list.
Each frame records:

- A unique frame ID and SHA-256 image checksum.
- A relative image path and verified pixel resolution.
- Source URL, source license, and redistribution-rights status.
- Recording group and match group.
- Overlay profile ID plus relative profile and catalog paths.
- Video frame timestamp in seconds.
- A fixed `development` or `holdout` partition.
- Whether the layout is expected to be supported.
- Verified blue and red teams plus ten unique role-aligned champion labels.
- The verifier identity and `verified` label status.

All paths must remain inside the manifest directory.
Recording groups and match groups may not cross partitions.
This prevents nearby animation frames or frames from the same game from leaking into both development and holdout evaluation.

## Capture guidance

Capture multiple frames from distinct recordings rather than many nearly identical frames from one draft.
Include every target resolution, compression level, language feed, animation phase, side swap, tournament, and overlay revision.
Include unsupported layouts so false automatic acceptance can be measured.
Avoid frames containing personal information or unrelated desktop content.

Use a development partition to define crop geometry, catalogs, similarity thresholds, and supported layouts.
Do not change those choices after inspecting holdout errors.
A new parser or threshold choice requires a new untouched grouped holdout.

One team or champion may have multiple verified portrait variants.
Every catalog entry identifies its variant, and the parser keeps only the best score for each canonical value before computing the runner-up margin.
This prevents two variants of the same champion from creating a falsely small confidence margin.

Some horizontal overlays display champion portraits in draft selection order while rendering player names in fixed role order.
Those columns must not be treated as player-champion assignments.
The detected champion set must be reconciled into verified top, jungle, mid, bottom, and support order before player-champion features are generated.

## Benchmark command

Run the benchmark with:

```bash
uv run lolpredictor vision-benchmark \
  --manifest data/raw/vision/lck-2026-v1/corpus.json \
  --output var/vision/lck-2026-v1-report.json
```

The command verifies every image checksum, resolution, profile ID, and path before measuring the parser.
It reports development and holdout metrics separately and breaks holdout results down by overlay profile.

## Fixed release gates

The initial real-overlay release requires:

- At least 100 supported holdout frames.
- At least three holdout recording groups.
- At least ten holdout match groups.
- At least 25 supported holdout frames for each promoted profile.
- At least 99.5 percent precision among automatically accepted team fields.
- At least 99.5 percent precision among automatically accepted champion fields.
- At least 95 percent exact ten-champion draft accuracy for each promoted profile.
- Zero automatic acceptances for labeled unsupported layouts.

Coverage, confirmation-required counts, and hard abstentions are reported even when precision passes.
A parser that accepts almost nothing cannot pass through precision alone because the size and exact-draft gates remain independent.

## Current status

The corpus contract, grouped split validation, integrity checks, metrics, abstention accounting, CLI, and CI fixture are implemented.
Three user-supplied real broadcast frames are available locally as development material, including full-height, vertically cropped, and player-stat-overlay captures.
They revealed side-specific team-logo variants, champion portrait variants, selection-order portrait rendering, and text occlusion over pick portraits.
No real frame is in a fixed grouped holdout yet.
Screenshot parsing is therefore not promoted for unattended use.

# Preliminary public-data benchmark

## Scope

This report records the first real-data research run for the version 2 pipeline.
It is a reproducible preliminary benchmark, not a current production release.
The measured source ends on November 2, 2024, so it is stale for predictions made in July 2026.

The source was the [Kaggle mirror of Oracle's Elixir professional-play data](https://www.kaggle.com/datasets/lauffing/oracles-elixir-league-of-legends-pro-play-data), retrieved on July 30, 2026.
Kaggle identifies that mirror as CC BY-NC-SA 4.0.
The direct [Oracle's Elixir download page](https://oracleselixir.com/tools/downloads) remains the preferred source for a current release.
The direct Google Drive files for 2025 and 2026 were temporarily quota-limited during this run, so they were not silently replaced with an unverified current source.

The downloaded archive SHA-256 was `f0bc1383c2e6f1cd2851d5cf17147d6e6d10b95c686bcdec322e15a6ffc53792`.
Its input workbooks were:

| Year | Workbook SHA-256 |
| --- | --- |
| 2020 | `c2e7107fa99db06f5e7750b346d03ea83710073cf181a48067351d9447880a3b` |
| 2021 | `e55af32114571066f77bd44fe9ae2f4e8f50ca5a949b24860c3b973691ba5b43` |
| 2022 | `f2e8ad478eb79a495b04747c83395ceff032f93a60e04197cecedf5ed5daa30b` |
| 2023 | `5220b87577d5a6aa2f73c86997a5d1c114d12e811695a946ede28867d8514f62` |
| 2024 | `42b48a868dd1644c60c15884db725c411acb1724d5059a78be6a092c4badd24a` |

Raw files, databases, reports, and model artifacts are ignored by Git.
The adapter hashes every original input and stores provenance in DuckDB.
Direct ingestion of the five workbooks produced canonical dataset fingerprint `d28e973d24a3893746653627080580f1b6b41e230dee079dd8a4be0c471255d9`.
That fingerprint exactly matched the measured benchmark database.

## Reproduction commands

Install the locked environment:

```bash
uv sync --locked --all-groups
```

Download and verify the exact research mirror:

```bash
mkdir -p data/raw/oe-2020-2024
curl -fL \
  -o data/raw/oe-2020-2024.zip \
  "https://www.kaggle.com/api/v1/datasets/download/lauffing/oracles-elixir-league-of-legends-pro-play-data"
sha256sum data/raw/oe-2020-2024.zip
unzip data/raw/oe-2020-2024.zip -d data/raw/oe-2020-2024
```

Ingest each original workbook through the strict pre-game allowlist:

```bash
for year in 2020 2021 2022 2023 2024; do
  uv run lolpredictor ingest-oe \
    --database var/public-oe-2020-2024/matches.duckdb \
    --input "data/raw/oe-2020-2024/pro_play_${year}.xlsx" \
    --source-url "https://www.kaggle.com/datasets/lauffing/oracles-elixir-league-of-legends-pro-play-data" \
    --retrieved-at "2026-07-30T18:00:00Z" \
    --legacy-blue-first-pick
done
```

The legacy first-pick flag is explicit because these pre-2026 workbooks do not contain Riot's independent First Selection field.
The adapter rejects use of that assumption for a source year after 2025.

Generate point-in-time features, run development backtests, and perform the release evaluation:

```bash
uv run lolpredictor features \
  --database var/public-oe-2020-2024/matches.duckdb \
  --config configs/real-baselines.yaml

uv run lolpredictor backtest \
  --database var/public-oe-2020-2024/matches.duckdb \
  --config configs/real-baselines.yaml \
  --reports var/public-oe-2020-2024/backtest-reports

uv run lolpredictor train \
  --database var/public-oe-2020-2024/matches.duckdb \
  --config configs/real-baselines.yaml \
  --registry var/public-oe-2020-2024/evaluation-artifacts \
  --reports var/public-oe-2020-2024/release-reports
```

## Data quality

The source contained 15,819 games.
The adapter accepted 15,627 games and quarantined 192 games instead of guessing.

| Year | Source games | Accepted | Quarantined |
| --- | ---: | ---: | ---: |
| 2020 | 2,956 | 2,946 | 10 |
| 2021 | 3,157 | 3,109 | 48 |
| 2022 | 3,280 | 3,256 | 24 |
| 2023 | 3,358 | 3,285 | 73 |
| 2024 | 3,068 | 3,031 | 37 |

Accepted games span January 13, 2020 through November 2, 2024.
They cover 175 stable team IDs, 8,670 reconstructed series, 74 patches, and 11 league labels.
The included league labels were LPL, LCK, VCS, PCS, LCS, LEC, CBLOL, LJL, LLA, WLDs, and MSI.

## Temporal evaluation

The fit and calibration data cutoff for candidate selection was January 22, 2023 at 16:24:46 UTC.
The 3,126-game validation interval ran from January 22, 2023 at 17:07:35 UTC through October 21, 2023.
The 3,126-game final holdout ran from October 22, 2023 through November 2, 2024.
The selected family was chosen by validation log loss only.

| Candidate | Validation log loss | Holdout log loss | Holdout Brier | Holdout ECE | Holdout ROC-AUC | Holdout accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blue-side base rate | 0.6909 | 0.6917 | 0.2493 | 0.0070 | 0.5000 | 52.82% |
| Elo only | 0.6224 | 0.6118 | 0.2124 | 0.0363 | 0.7234 | 65.80% |
| Team and roster logistic | 0.6142 | 0.6074 | 0.2100 | 0.0196 | 0.7302 | 67.08% |
| Draft-only logistic | 0.6858 | 0.6866 | 0.2468 | 0.0288 | 0.5655 | 54.96% |
| Combined logistic | 0.6136 | 0.6072 | 0.2095 | 0.0356 | 0.7354 | 67.47% |
| Gradient-boosted trees | 0.6205 | 0.6127 | 0.2122 | 0.0298 | 0.7248 | 66.28% |
| 30% Elo plus 70% combined logistic | **0.6125** | **0.6027** | **0.2080** | 0.0222 | **0.7363** | **67.69%** |

The selected blend's series-clustered 95 percent holdout log-loss interval was 0.5873 to 0.6176 across 1,464 series clusters.
Across 3,749 rolling out-of-time development predictions, the fixed blend produced 0.6178 log loss, 0.2143 Brier score, 0.0320 ECE, 0.7173 ROC-AUC, and 66.02 percent accuracy.
The rolling backtest did not evaluate the reserved final holdout.

After evaluation, the production refit used 14,377 games through June 19, 2024 for estimator fitting and 1,250 later games for chronological probability calibration.
Its feature state ends at November 2, 2024 at 17:53:48 UTC.
That production refit is useful for artifact-path verification but is too stale for a July 2026 prediction.

The holdout has now been examined.
It must not be reused as an untouched release gate for model changes.
A current 2025-2026 dataset and a newly reserved chronological holdout are required for the next release claim.

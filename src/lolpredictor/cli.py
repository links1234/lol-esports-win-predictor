"""Command-line entrypoint for the reproducible vertical slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from lolpredictor.fixtures import build_fixture_database
from lolpredictor.ingestion import ingest_jsonl
from lolpredictor.leaguepedia import (
    fetch_leaguepedia_snapshot,
    ingest_leaguepedia_snapshot,
)
from lolpredictor.optimization.refit import refit_locked_finalist
from lolpredictor.optimization.runner import optimization_status, run_optimization
from lolpredictor.oracles_elixir import ingest_oracles_elixir_source
from lolpredictor.prediction import predict_file, predict_jsonl
from lolpredictor.promotion import write_release_promotion_report
from lolpredictor.settings import load_settings
from lolpredictor.state_space.refit import refit_v6_finalist
from lolpredictor.state_space.runner import run_v6_study
from lolpredictor.training import (
    backtest_baselines,
    confirm_backtest_selection,
    evaluate_saved_artifact,
    materialize_historical_features,
    refit_development_artifact,
    refit_selected_artifact,
    train_baselines,
)
from lolpredictor.vision.benchmark import write_overlay_benchmark_report
from lolpredictor.vision.fixtures import build_screenshot_fixture
from lolpredictor.vision.parser import (
    load_screenshot_context,
    load_screenshot_corrections,
    parse_screenshot,
    predict_screenshot,
    write_json,
)

DEFAULT_CONFIG = Path("configs/baselines.yaml")
DEFAULT_SAMPLE = Path("examples/sample_draft.json")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _metric_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "roc_auc",
            "accuracy",
            "sample_count",
        )
    }


def _training_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_directory": report["artifact_directory"],
        "data_cutoff_timestamp": report["data_cutoff_timestamp"],
        "selection": report["selection"],
        "split_summary": report["split_summary"],
        "candidates": {
            name: {
                "validation": _metric_summary(metrics["validation"]),
                "test": _metric_summary(metrics["test"]),
            }
            for name, metrics in report["candidates"].items()
        },
    }


def _evaluation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": report["model_version"],
        "model_kind": report["model_kind"],
        "data_cutoff_timestamp": report["data_cutoff_timestamp"],
        "test": _metric_summary(report["test"]),
    }


def _backtest_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold_count": report["fold_count"],
        "aggregate_candidates": {
            name: _metric_summary(metrics)
            for name, metrics in report["aggregate_candidates"].items()
        },
        "aggregate_selection": report["aggregate_selection"],
        "selection_policy": {
            "selected_candidate_counts": report["selection_policy"]["selected_candidate_counts"],
            "test": _metric_summary(report["selection_policy"]["test"]),
        },
        "reserved_final_holdout": report["reserved_final_holdout"],
        "reserved_final_holdout_evaluated": report["reserved_final_holdout_evaluated"],
    }


def _confirmation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection": report["selection"],
        "training_data_cutoff_timestamp": report["training_data_cutoff_timestamp"],
        "confirmation_status": report["confirmation_status"],
        "candidate_validation": _metric_summary(report["candidate_validation"]),
        "elo_validation": _metric_summary(report["elo_validation"]),
        "reserved_final_holdout": report["reserved_final_holdout"],
        "reserved_final_holdout_evaluated": report["reserved_final_holdout_evaluated"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lolpredictor",
        description="Leakage-safe League of Legends draft win prediction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture", help="Build the synthetic fixture DuckDB")
    fixture.add_argument("--database", type=Path, required=True)
    fixture.add_argument("--force", action="store_true")

    ingest = subparsers.add_parser("ingest", help="Validate and ingest historical JSONL")
    ingest.add_argument("--database", type=Path, required=True)
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--source", required=True)

    ingest_oe = subparsers.add_parser(
        "ingest-oe",
        help="Validate and ingest a pinned Oracle's Elixir CSV or XLSX file",
    )
    ingest_oe.add_argument("--database", type=Path, required=True)
    ingest_oe.add_argument("--input", type=Path, required=True)
    ingest_oe.add_argument("--source-url")
    ingest_oe.add_argument(
        "--retrieved-at",
        type=datetime.fromisoformat,
        help="Timezone-aware source retrieval timestamp",
    )
    ingest_oe.add_argument(
        "--legacy-blue-first-pick",
        action="store_true",
        help="For pre-2026 files lacking firstPick, derive it from blue side",
    )

    fetch_leaguepedia = subparsers.add_parser(
        "fetch-leaguepedia",
        help="Fetch a pinned Leaguepedia pre-game snapshot",
    )
    fetch_leaguepedia.add_argument("--output", type=Path, required=True)
    fetch_leaguepedia.add_argument(
        "--start",
        type=datetime.fromisoformat,
        required=True,
        help="Timezone-aware inclusive interval start",
    )
    fetch_leaguepedia.add_argument(
        "--end",
        type=datetime.fromisoformat,
        required=True,
        help="Timezone-aware exclusive interval end",
    )
    fetch_leaguepedia.add_argument(
        "--retrieved-at",
        type=datetime.fromisoformat,
        help="Pinned timezone-aware retrieval timestamp",
    )
    fetch_leaguepedia.add_argument("--page-size", type=int, default=5000)

    ingest_leaguepedia = subparsers.add_parser(
        "ingest-leaguepedia",
        help="Validate and ingest a pinned Leaguepedia snapshot",
    )
    ingest_leaguepedia.add_argument("--database", type=Path, required=True)
    ingest_leaguepedia.add_argument("--input", type=Path, required=True)

    features = subparsers.add_parser(
        "features", help="Generate strict point-in-time historical features"
    )
    features.add_argument("--database", type=Path, required=True)
    features.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    train = subparsers.add_parser("train", help="Train and select all baseline models")
    train.add_argument("--database", type=Path, required=True)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--registry", type=Path, required=True)
    train.add_argument("--reports", type=Path, required=True)

    backtest = subparsers.add_parser(
        "backtest",
        help="Run rolling-origin baseline backtests while reserving the final holdout",
    )
    backtest.add_argument("--database", type=Path, required=True)
    backtest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    backtest.add_argument("--reports", type=Path, required=True)

    optimize = subparsers.add_parser(
        "optimize",
        help="Run or resume the frozen nested chronological model search",
    )
    optimize.add_argument("--database", type=Path, required=True)
    optimize.add_argument("--config", type=Path, required=True)
    optimize.add_argument("--study", type=Path, required=True)

    optimize_status = subparsers.add_parser(
        "optimize-status",
        help="Read the latest optimizer status without changing the study",
    )
    optimize_status.add_argument("--study", type=Path, required=True)

    optimize_refit = subparsers.add_parser(
        "optimize-refit",
        help="Refit a locked optimizer finalist as a development artifact",
    )
    optimize_refit.add_argument("--database", type=Path, required=True)
    optimize_refit.add_argument("--config", type=Path, required=True)
    optimize_refit.add_argument("--locked-finalist", type=Path, required=True)
    optimize_refit.add_argument("--nested-report", type=Path, required=True)
    optimize_refit.add_argument("--registry", type=Path, required=True)
    optimize_refit.add_argument("--output", type=Path, required=True)

    state_space_study = subparsers.add_parser(
        "state-space-study",
        help="Run the preregistered v6 hierarchical state-space comparison",
    )
    state_space_study.add_argument("--database", type=Path, required=True)
    state_space_study.add_argument("--config", type=Path, required=True)
    state_space_study.add_argument("--study", type=Path, required=True)

    state_space_refit = subparsers.add_parser(
        "state-space-refit",
        help="Refit a locked v6 finalist as a development artifact",
    )
    state_space_refit.add_argument("--database", type=Path, required=True)
    state_space_refit.add_argument("--config", type=Path, required=True)
    state_space_refit.add_argument("--locked-finalist", type=Path, required=True)
    state_space_refit.add_argument("--study-report", type=Path, required=True)
    state_space_refit.add_argument("--registry", type=Path, required=True)
    state_space_refit.add_argument("--output", type=Path, required=True)

    confirm_selection = subparsers.add_parser(
        "confirm-selection",
        help="Evaluate the locked backtest winner on validation only",
    )
    confirm_selection.add_argument("--database", type=Path, required=True)
    confirm_selection.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    confirm_selection.add_argument("--backtest-report", type=Path, required=True)
    confirm_selection.add_argument("--output", type=Path, required=True)

    release_gate = subparsers.add_parser(
        "release-gate",
        help="Apply preregistered promotion gates to training and backtest reports",
    )
    release_gate.add_argument("--config", type=Path, required=True)
    release_gate.add_argument("--training-report", type=Path, required=True)
    release_gate.add_argument("--backtest-report", type=Path, required=True)
    release_gate.add_argument("--output", type=Path, required=True)

    refit = subparsers.add_parser(
        "refit",
        help="Refit the evaluated winner and snapshot state through all available data",
    )
    refit.add_argument("--database", type=Path, required=True)
    refit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    refit.add_argument("--training-report", type=Path, required=True)
    refit.add_argument("--registry", type=Path, required=True)
    refit.add_argument("--reports", type=Path, required=True)
    refit.add_argument(
        "--promotion-report",
        type=Path,
        help="Use the gate-recommended candidate, including an Elo fallback",
    )

    refit_development = subparsers.add_parser(
        "refit-development",
        help="Build a non-promoted shadow artifact from the locked backtest selection",
    )
    refit_development.add_argument("--database", type=Path, required=True)
    refit_development.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    refit_development.add_argument("--backtest-report", type=Path, required=True)
    refit_development.add_argument(
        "--confirmation-report",
        type=Path,
        required=True,
    )
    refit_development.add_argument("--registry", type=Path, required=True)
    refit_development.add_argument("--reports", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a saved artifact on its chronological holdout"
    )
    evaluate.add_argument("--database", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    predict = subparsers.add_parser("predict", help="Predict one draft JSON")
    predict.add_argument("--artifact", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)

    predict_batch = subparsers.add_parser(
        "predict-batch", help="Predict newline-delimited draft JSON"
    )
    predict_batch.add_argument("--artifact", type=Path, required=True)
    predict_batch.add_argument("--input", type=Path, required=True)
    predict_batch.add_argument("--output", type=Path, required=True)

    vision_fixture = subparsers.add_parser(
        "vision-fixture",
        help="Build the deterministic supported-overlay screenshot fixture",
    )
    vision_fixture.add_argument("--directory", type=Path, required=True)
    vision_fixture.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)

    parse_image = subparsers.add_parser(
        "parse-image",
        help="Extract a draft candidate from a supported broadcast screenshot",
    )
    parse_image.add_argument("--input", type=Path, required=True)
    parse_image.add_argument("--profile", type=Path, required=True)
    parse_image.add_argument("--catalog", type=Path, required=True)
    parse_image.add_argument("--output", type=Path)

    predict_image = subparsers.add_parser(
        "predict-image",
        help="Parse, confirm, and predict a supported broadcast screenshot",
    )
    predict_image.add_argument("--artifact", type=Path, required=True)
    predict_image.add_argument("--input", type=Path, required=True)
    predict_image.add_argument("--profile", type=Path, required=True)
    predict_image.add_argument("--catalog", type=Path, required=True)
    predict_image.add_argument("--context", type=Path, required=True)
    predict_image.add_argument("--corrections", type=Path)
    predict_image.add_argument("--output", type=Path)

    vision_benchmark = subparsers.add_parser(
        "vision-benchmark",
        help="Benchmark a verified real-overlay corpus on grouped holdouts",
    )
    vision_benchmark.add_argument("--manifest", type=Path, required=True)
    vision_benchmark.add_argument("--output", type=Path, required=True)

    run_all = subparsers.add_parser("run-all", help="Run the complete synthetic vertical slice")
    run_all.add_argument("--workdir", type=Path, required=True)
    run_all.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_all.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    return parser


def _run_all(workdir: Path, config_path: Path, sample_path: Path) -> dict[str, Any]:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    database = workdir / "fixture.duckdb"
    registry = workdir / "artifacts"
    reports = workdir / "reports"
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run-all database: {database}")

    fixture = build_fixture_database(database)
    settings = load_settings(config_path)
    features = materialize_historical_features(database, settings)
    backtest = backtest_baselines(
        database,
        settings,
        reports_directory=reports / "backtest",
    )
    training = train_baselines(
        database,
        settings,
        registry_directory=registry,
        reports_directory=reports,
    )
    artifact = Path(training["artifact_directory"])
    evaluation = evaluate_saved_artifact(
        database,
        artifact,
        output_path=reports / "evaluation.json",
    )
    prediction = predict_file(artifact, sample_path).model_dump(mode="json")
    (reports / "sample-prediction.json").write_text(
        json.dumps(prediction, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    vision_fixture = build_screenshot_fixture(workdir / "vision", sample_path)
    screenshot_prediction = predict_screenshot(
        artifact,
        Path(vision_fixture["screenshot"]),
        Path(vision_fixture["profile"]),
        Path(vision_fixture["catalog"]),
        load_screenshot_context(Path(vision_fixture["context"])),
    )
    write_json(reports / "sample-screenshot-prediction.json", screenshot_prediction)
    vision_benchmark = write_overlay_benchmark_report(
        Path(vision_fixture["manifest"]),
        reports / "vision-benchmark.json",
    )
    return {
        "fixture": fixture,
        "features": features,
        "backtest": _backtest_summary(backtest),
        "training": _training_summary(training),
        "evaluation": _evaluation_summary(evaluation),
        "prediction": prediction,
        "vision_fixture": vision_fixture,
        "vision_benchmark": vision_benchmark,
        "screenshot_prediction": screenshot_prediction,
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "fixture":
        _print_json(build_fixture_database(args.database, force=args.force))
    elif args.command == "ingest":
        count = ingest_jsonl(args.database, args.input, source=args.source)
        _print_json({"database": str(args.database.resolve()), "inserted": count})
    elif args.command == "ingest-oe":
        _print_json(
            ingest_oracles_elixir_source(
                args.database,
                args.input,
                source_url=args.source_url,
                retrieved_at=args.retrieved_at,
                assume_legacy_blue_first_pick=args.legacy_blue_first_pick,
            )
        )
    elif args.command == "fetch-leaguepedia":
        _print_json(
            fetch_leaguepedia_snapshot(
                args.output,
                start_timestamp=args.start,
                end_timestamp=args.end,
                retrieved_at=args.retrieved_at,
                page_size=args.page_size,
            )
        )
    elif args.command == "ingest-leaguepedia":
        _print_json(
            ingest_leaguepedia_snapshot(
                args.database,
                args.input,
            )
        )
    elif args.command == "features":
        _print_json(
            materialize_historical_features(
                args.database,
                load_settings(args.config),
            )
        )
    elif args.command == "train":
        report = train_baselines(
            args.database,
            load_settings(args.config),
            registry_directory=args.registry,
            reports_directory=args.reports,
        )
        _print_json(_training_summary(report))
    elif args.command == "backtest":
        report = backtest_baselines(
            args.database,
            load_settings(args.config),
            reports_directory=args.reports,
        )
        _print_json(_backtest_summary(report))
    elif args.command == "optimize":
        _print_json(
            run_optimization(
                args.database,
                args.config,
                args.study,
            )
        )
    elif args.command == "optimize-status":
        _print_json(optimization_status(args.study))
    elif args.command == "optimize-refit":
        _print_json(
            refit_locked_finalist(
                args.database,
                args.config,
                locked_finalist_path=args.locked_finalist,
                nested_report_path=args.nested_report,
                registry_directory=args.registry,
                output_path=args.output,
            )
        )
    elif args.command == "state-space-study":
        _print_json(
            run_v6_study(
                args.database,
                args.config,
                args.study,
            )
        )
    elif args.command == "state-space-refit":
        _print_json(
            refit_v6_finalist(
                args.database,
                args.config,
                locked_finalist_path=args.locked_finalist,
                study_report_path=args.study_report,
                registry_directory=args.registry,
                output_path=args.output,
            )
        )
    elif args.command == "confirm-selection":
        report = confirm_backtest_selection(
            args.database,
            load_settings(args.config),
            backtest_report_path=args.backtest_report,
            output_path=args.output,
        )
        _print_json(_confirmation_summary(report))
    elif args.command == "release-gate":
        _print_json(
            write_release_promotion_report(
                args.training_report,
                args.backtest_report,
                load_settings(args.config),
                args.output,
            )
        )
    elif args.command == "refit":
        report = refit_selected_artifact(
            args.database,
            load_settings(args.config),
            training_report_path=args.training_report,
            registry_directory=args.registry,
            reports_directory=args.reports,
            promotion_report_path=args.promotion_report,
        )
        _print_json(
            {
                "artifact_directory": report["artifact_directory"],
                "artifact_purpose": report["artifact_purpose"],
                "model_kind": report["model_kind"],
                "data_cutoff_timestamp": report["data_cutoff_timestamp"],
                "split_summary": report["split_summary"],
            }
        )
    elif args.command == "refit-development":
        report = refit_development_artifact(
            args.database,
            load_settings(args.config),
            backtest_report_path=args.backtest_report,
            confirmation_report_path=args.confirmation_report,
            registry_directory=args.registry,
            reports_directory=args.reports,
        )
        _print_json(
            {
                "artifact_directory": report["artifact_directory"],
                "artifact_purpose": report["artifact_purpose"],
                "model_kind": report["model_kind"],
                "data_cutoff_timestamp": report["data_cutoff_timestamp"],
                "prospective_promotion_passed": report["prospective_promotion_passed"],
            }
        )
    elif args.command == "evaluate":
        report = evaluate_saved_artifact(
            args.database,
            args.artifact,
            output_path=args.output,
        )
        _print_json(_evaluation_summary(report))
    elif args.command == "predict":
        response = predict_file(args.artifact, args.input)
        payload = response.model_dump(mode="json")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        _print_json(payload)
    elif args.command == "predict-batch":
        _print_json(predict_jsonl(args.artifact, args.input, args.output))
    elif args.command == "vision-fixture":
        _print_json(build_screenshot_fixture(args.directory, args.sample))
    elif args.command == "parse-image":
        payload = parse_screenshot(
            args.input,
            args.profile,
            args.catalog,
        ).model_dump(mode="json")
        if args.output:
            write_json(args.output, payload)
        _print_json(payload)
    elif args.command == "predict-image":
        corrections = load_screenshot_corrections(args.corrections) if args.corrections else None
        payload = predict_screenshot(
            args.artifact,
            args.input,
            args.profile,
            args.catalog,
            load_screenshot_context(args.context),
            corrections,
        )
        if args.output:
            write_json(args.output, payload)
        _print_json(payload)
    elif args.command == "vision-benchmark":
        _print_json(
            write_overlay_benchmark_report(
                args.manifest,
                args.output,
            )
        )
    elif args.command == "run-all":
        _print_json(_run_all(args.workdir, args.config, args.sample))
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

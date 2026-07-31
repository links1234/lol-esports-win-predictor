import ast
import re
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    paths = [REPOSITORY_ROOT / name.decode() for name in result.stdout.split(b"\0") if name]
    return [path for path in paths if path.exists()]


def test_generated_state_and_credentials_are_not_tracked() -> None:
    relative_paths = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in tracked_files()}
    forbidden_suffixes = (".duckdb", ".keras", ".h5", ".joblib", ".pkl", ".parquet")
    assert ".env" not in relative_paths
    assert not any(path.startswith("venv/") for path in relative_paths)
    assert not any(path.startswith(".venv/") for path in relative_paths)
    assert not any(path.endswith(forbidden_suffixes) for path in relative_paths)
    assert not any(path.startswith("mlruns/") for path in relative_paths)


def test_no_credential_pattern_or_em_dash_is_present() -> None:
    secret_pattern = re.compile(rb"RGAPI-[A-Za-z0-9_-]{16,}")
    for path in tracked_files():
        if not path.is_file():
            continue
        content = path.read_bytes()
        assert not secret_pattern.search(content), path
        if b"\0" not in content:
            assert "\N{EM DASH}" not in content.decode("utf-8"), path


def test_source_never_calls_eval_or_exec() -> None:
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dangerous = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
        ]
        assert not dangerous, path


def test_final_modeling_code_has_no_random_train_test_split() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (REPOSITORY_ROOT / "src").rglob("*.py")
    )
    assert "train_test_split" not in source

from pathlib import Path


def test_numpy_is_pinned_per_execution_platform() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '\'numpy==1.26.4; platform_system == "Darwin"\'' in pyproject
    assert '\'numpy==2.2.6; platform_system == "Linux"\'' in pyproject
    assert '\n  "numpy==1.26.4",' not in pyproject

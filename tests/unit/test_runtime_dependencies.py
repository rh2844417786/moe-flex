from pathlib import Path

import tomllib


def test_numpy_is_pinned_per_execution_platform() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    dependencies = set(project["dependencies"])

    assert 'numpy==1.26.4; platform_system == "Darwin"' in dependencies
    assert 'numpy==2.2.6; platform_system == "Linux"' in dependencies
    assert "numpy==1.26.4" not in dependencies

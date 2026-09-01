import subprocess
from pathlib import Path


def test_server_scripts_have_valid_bash_syntax() -> None:
    scripts = sorted(Path("scripts/server").glob("*.sh"))

    assert len(scripts) == 5
    for script in scripts:
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_container_script_enforces_mount_and_isolation_contract() -> None:
    script = Path("scripts/server/run_container.sh").read_text(encoding="utf-8")

    assert "/home/jovyan/wangtonghan/moe-flex" in script
    assert "src=/mnt/public_data,dst=/mnt/public_data,readonly" in script
    assert "--ipc=host" in script
    assert "--entrypoint" in script
    assert "wth333" not in script


def test_image_build_uses_exact_checkout_sha_not_latest() -> None:
    script = Path("scripts/server/build.sh").read_text(encoding="utf-8")

    assert "git_sha=" in script
    assert "ghcr.io/rh2844417786/moe-flex:${git_sha}" in script
    assert ":latest" not in script

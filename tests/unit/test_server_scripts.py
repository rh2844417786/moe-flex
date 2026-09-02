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


def test_image_build_uses_docker_hub_base_and_local_sha_tag() -> None:
    build_script = Path("scripts/server/build.sh").read_text(encoding="utf-8")
    run_script = Path("scripts/server/run_container.sh").read_text(encoding="utf-8")

    base_digest = (
        "vllm/vllm-openai:v0.10.2@"
        "sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db"
    )
    assert f'base_image="{base_digest}"' in build_script
    assert 'image="moe-flex-local:${git_sha}"' in build_script
    assert 'docker pull "${base_image}"' in build_script
    assert "--network=none" in build_script
    assert "ghcr.io" not in build_script
    assert ":latest" not in build_script
    assert 'source "${project_root}/build/image.env"' in run_script
    assert "ghcr.io" not in run_script


def test_server_dockerfile_has_no_non_dockerhub_network_steps() -> None:
    dockerfile = Path("docker/Dockerfile").read_text(encoding="utf-8")

    assert (
        "FROM vllm/vllm-openai:v0.10.2@"
        "sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db"
        in dockerfile
    )
    assert "apt-get" not in dockerfile
    assert "git clone" not in dockerfile
    assert "--no-deps" in dockerfile
    assert "--no-build-isolation" in dockerfile
    assert "--no-index" in dockerfile

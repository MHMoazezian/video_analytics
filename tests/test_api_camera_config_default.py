import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
import yaml

from app.api import main
from app.api.jobs import JobManager
from app.geometry.config import load_camera_config


def test_empty_camera_config_upload_is_treated_as_missing() -> None:
    upload = UploadFile(file=BytesIO(), filename="", size=0)

    assert not main._has_camera_config_upload(upload)


def test_named_camera_config_upload_is_present() -> None:
    upload = UploadFile(file=BytesIO(b"camera: {}"), filename="camera.yaml", size=10)

    assert main._has_camera_config_upload(upload)


def test_bundled_example_is_copied_when_no_saved_default_exists(
    tmp_path: Path, monkeypatch,
) -> None:
    saved = tmp_path / "settings" / "camera.yaml"
    destination = tmp_path / "job" / "camera.yaml"
    destination.parent.mkdir()
    monkeypatch.setattr(main, "SAVED_CAMERA_CONFIG_PATH", saved)

    main._copy_default_camera_config(destination)

    assert load_camera_config(destination) == load_camera_config(
        main.DEFAULT_CAMERA_CONFIG_PATH
    )


def test_uploaded_config_becomes_the_saved_default(
    tmp_path: Path, monkeypatch,
) -> None:
    saved = tmp_path / "settings" / "camera.yaml"
    destination = tmp_path / "next-job" / "camera.yaml"
    destination.parent.mkdir()
    monkeypatch.setattr(main, "SAVED_CAMERA_CONFIG_PATH", saved)

    main._save_default_camera_config(main.DEFAULT_CAMERA_CONFIG_PATH)
    main._copy_default_camera_config(destination)

    assert saved.is_file()
    assert load_camera_config(destination) == load_camera_config(saved)


def _custom_camera_yaml() -> bytes:
    """The bundled example under another camera id, so the two are distinguishable."""

    mapping = yaml.safe_load(main.DEFAULT_CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
    mapping["camera"]["id"] = "saved-default-camera"
    return yaml.safe_dump(mapping, allow_unicode=True).encode("utf-8")


def _submit_job(camera_yaml: bytes | None, *, persist: bool = True) -> dict[str, object]:
    video = UploadFile(file=BytesIO(b"not-a-real-video"), filename="shop.mp4", size=16)
    # An untouched browser file picker still submits an empty, unnamed part.
    camera_config = (
        UploadFile(file=BytesIO(), filename="", size=0)
        if camera_yaml is None
        else UploadFile(file=BytesIO(camera_yaml), filename="camera.yaml", size=len(camera_yaml))
    )
    response = asyncio.run(
        main.create_job(
            video=video,
            application_id="people_counting",
            camera_id="uploaded-video",
            max_frames=None,
            enable_reid=False,
            tracker_type="bytetrack",
            camera_config=camera_config,
            persist_camera_config=persist,
        )
    )
    return response["data"]  # type: ignore[return-value]


def _isolated_manager(tmp_path: Path, monkeypatch) -> tuple[JobManager, Path]:
    manager = JobManager(tmp_path / "jobs")
    monkeypatch.setattr(manager, "enqueue", lambda _job_id: None)
    saved = tmp_path / "jobs" / "_settings" / "camera.yaml"
    monkeypatch.setattr(main, "manager", manager)
    monkeypatch.setattr(main, "SAVED_CAMERA_CONFIG_PATH", saved)
    return manager, saved


def test_job_without_upload_keeps_and_reuses_the_saved_default(
    tmp_path: Path, monkeypatch,
) -> None:
    manager, saved = _isolated_manager(tmp_path, monkeypatch)

    first = _submit_job(_custom_camera_yaml())
    assert load_camera_config(saved).camera_id == "saved-default-camera"
    saved_bytes = saved.read_bytes()

    second = _submit_job(None)

    # The saved default is neither clobbered by the bundled example...
    assert saved.read_bytes() == saved_bytes
    # ...and it is what the job without an upload actually runs with.
    job_config = Path(manager.get(str(second["id"])).camera_config or "")
    assert load_camera_config(job_config).camera_id == "saved-default-camera"
    assert first["id"] != second["id"]


def test_job_without_upload_uses_the_bundled_example_without_saving_it(
    tmp_path: Path, monkeypatch,
) -> None:
    manager, saved = _isolated_manager(tmp_path, monkeypatch)

    job = _submit_job(None)

    assert not saved.exists()
    job_config = Path(manager.get(str(job["id"])).camera_config or "")
    assert load_camera_config(job_config) == load_camera_config(main.DEFAULT_CAMERA_CONFIG_PATH)


def test_upload_is_not_persisted_when_the_caller_opts_out(
    tmp_path: Path, monkeypatch,
) -> None:
    manager, saved = _isolated_manager(tmp_path, monkeypatch)

    job = _submit_job(_custom_camera_yaml(), persist=False)

    assert not saved.exists()
    job_config = Path(manager.get(str(job["id"])).camera_config or "")
    assert load_camera_config(job_config).camera_id == "saved-default-camera"


def test_default_camera_config_constant_is_not_redefined() -> None:
    from app.core import config

    assert main.DEFAULT_CAMERA_CONFIG_PATH is config.DEFAULT_CAMERA_CONFIG_PATH

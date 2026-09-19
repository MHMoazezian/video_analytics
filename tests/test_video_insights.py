from __future__ import annotations

import cv2
import numpy as np
from contextlib import nullcontext

from app.insights.service import (
    FRUIT_QUALITY_PROMPT,
    VideoInsightError,
    VideoInsightService,
    _parse_fruit_quality,
    _parse_yes_no_answers,
    extract_stream_frames,
    extract_video_frames,
)


class _SeekableCapture:
    def __init__(self, total: int = 12, fps: float = 10.0) -> None:
        self.total = total
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total)
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        # BGR values make the color conversion observable.
        return True, np.full((4, 6, 3), [self.index, 10, 20], dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_extract_video_frames_samples_uniformly_and_converts_to_rgb(monkeypatch) -> None:
    capture = _SeekableCapture(total=12)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    frames = extract_video_frames("sample.mp4", num_frames=4)

    assert len(frames) == 4
    assert [int(frame[0, 0, 2]) for frame in frames] == [0, 3, 7, 11]
    assert frames[0][0, 0].tolist() == [20, 10, 0]
    assert capture.released


def test_extract_video_frames_samples_only_requested_time_window(monkeypatch) -> None:
    capture = _SeekableCapture(total=120, fps=10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    frames = extract_video_frames(
        "sample.mp4",
        num_frames=4,
        window_start_seconds=2,
        window_end_seconds=5,
    )

    assert [int(frame[0, 0, 2]) for frame in frames] == [20, 29, 39, 49]


def test_extract_stream_frames_returns_consecutive_frames(monkeypatch) -> None:
    capture = _SeekableCapture()

    def read() -> tuple[bool, np.ndarray]:
        capture.index += 1
        return True, np.full((4, 6, 3), capture.index, dtype=np.uint8)

    capture.read = read  # type: ignore[method-assign]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _url: capture)

    frames = extract_stream_frames(
        "rtsp://mediamtx:8554/camera", num_frames=3, sample_interval_seconds=0
    )

    assert len(frames) == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [1, 2, 3]
    assert capture.released


def test_interpret_returns_only_normalized_yes_no_answers(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    calls: list[dict[str, object]] = []

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return ('```json\n{"fighting": "yes", "floor_clean": false}\n```', 5.3)

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret([np.zeros((4, 6, 3), dtype=np.uint8)] * 8)

    assert result["answers"] == {"fighting": "Yes", "floor_clean": "No"}
    assert result["inference_seconds"] == 5.3
    assert set(result) == {"answers", "frame_count", "inference_seconds"}
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 40
    assert len(calls[0]["images"]) == 8  # type: ignore[arg-type]


def test_answer_parser_rejects_free_form_model_output() -> None:
    with np.testing.assert_raises(VideoInsightError):
        _parse_yes_no_answers("There does not seem to be a fight.")


def test_interpret_prompt_contains_both_fixed_questions(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    prompts: list[object] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3])
        return ('{"fighting":"No","floor_clean":"Yes"}', 1.0)

    monkeypatch.setattr(service, "_generate", fake_generate)
    service.interpret([np.zeros((4, 6, 3), dtype=np.uint8)] * 8)

    visual_content = prompts[0][0]["content"]  # type: ignore[index]
    assert "Are there persons fighting in the video?" in visual_content[-1]["text"]
    assert "Is the floor of the scene clean?" in visual_content[-1]["text"]


def test_parse_fruit_quality_returns_normalized_contract() -> None:
    result = _parse_fruit_quality(
        '```json\n{"has_fruit":true,"label":"تقریباً تازه",'
        '"freshness_score":78,"distribution":{"fresh":75,"middle":20,"rotten":5},'
        '"fruit_count_estimate":9,"confidence":84,"summary_fa":"بیشتر میوه‌ها تازه هستند."}\n```'
    )

    assert result["label"] == "تقریباً تازه"
    assert result["distribution"] == {"fresh": 75, "middle": 20, "rotten": 5}
    assert result["freshness_score"] == 78


def test_parse_fruit_quality_rejects_invalid_distribution() -> None:
    with np.testing.assert_raises(VideoInsightError):
        _parse_fruit_quality(
            '{"has_fruit":true,"label":"متوسط","freshness_score":50,'
            '"distribution":{"fresh":40,"middle":40,"rotten":40},'
            '"fruit_count_estimate":3,"confidence":70,"summary_fa":"کیفیت متوسط است."}'
        )


def test_interpret_fruit_quality_uses_common_prompt(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    prompts: list[object] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3])
        return (
            '{"has_fruit":true,"label":"تازه","freshness_score":96,'
            '"distribution":{"fresh":100,"middle":0,"rotten":0},'
            '"fruit_count_estimate":4,"confidence":92,"summary_fa":"همه میوه‌ها تازه‌اند."}',
            1.2,
        )

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality([np.zeros((4, 6, 3), dtype=np.uint8)])

    assert result["label"] == "تازه"
    assert result["frame_count"] == 1
    assert prompts[0][0]["content"][-1]["text"] == FRUIT_QUALITY_PROMPT  # type: ignore[index]

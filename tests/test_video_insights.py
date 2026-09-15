from __future__ import annotations

import cv2
import numpy as np
from contextlib import nullcontext

from app.insights.service import VideoInsightService, extract_stream_frames, extract_video_frames


class _SeekableCapture:
    def __init__(self, total: int = 12) -> None:
        self.total = total
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return float(self.total) if prop == cv2.CAP_PROP_FRAME_COUNT else 0.0

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


def test_persian_query_is_translated_before_english_reasoning(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    calls: list[dict[str, object]] = []
    responses = iter([
        ("What are the people doing?", 0.2),
        ("They are walking through a market.", 5.3),
        ("آن‌ها در حال عبور از بازار هستند.", 0.4),
    ])

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret(
        [np.zeros((4, 6, 3), dtype=np.uint8)] * 8,
        "افراد چه کاری انجام می‌دهند؟",
    )

    assert result["translated_query"] == "What are the people doing?"
    assert result["english_text"] == "They are walking through a market."
    assert result["text"] == "آن‌ها در حال عبور از بازار هستند."
    assert result["inference_seconds"] == 5.3
    assert result["translation_seconds"] == 0.6
    assert [call["max_new_tokens"] for call in calls] == [64, 80, 160]
    assert calls[0].get("images") is None
    assert len(calls[1]["images"]) == 8  # type: ignore[arg-type]


def test_english_query_skips_input_translation(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    calls: list[dict[str, object]] = []
    responses = iter([
        ("No abnormal activity is visible.", 5.0),
        ("رفتار غیرعادی مشاهده نمی‌شود.", 0.3),
    ])

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret(
        [np.zeros((4, 6, 3), dtype=np.uint8)] * 8,
        "Is there abnormal behavior?",
    )

    assert result["translated_query"] == "Is there abnormal behavior?"
    assert len(calls) == 2
    assert calls[0]["images"] is not None


def test_detailed_mode_adds_chronological_evidence_instructions(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    prompts: list[object] = []
    responses = iter([
        ("A detailed English answer.", 5.0),
        ("پاسخ فارسی با جزئیات.", 0.3),
    ])

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3])
        return next(responses)

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret(
        [np.zeros((4, 6, 3), dtype=np.uint8)] * 8,
        "Describe the scene",
        detailed=True,
    )

    visual_content = prompts[0][0]["content"]  # type: ignore[index]
    assert "detailed chronological analysis" in visual_content[-1]["text"]
    assert result["detailed"] is True

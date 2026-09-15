from __future__ import annotations

import cv2
import numpy as np

from app.insights.service import extract_stream_frames, extract_video_frames


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

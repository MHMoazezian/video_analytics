"""Optimized Qwen2.5-VL inference for recorded videos and RTSP sequences."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np


class VideoInsightError(RuntimeError):
    """A user-facing video interpretation failure."""


def extract_video_frames(video_path: str | Path, *, num_frames: int = 8) -> list[np.ndarray]:
    """Return evenly spaced RGB frames without decoding the entire file."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoInsightError("video does not contain any decodable frames")
        count = min(num_frames, total_frames)
        indexes = np.linspace(0, total_frames - 1, num=count, dtype=int)
        frames: list[np.ndarray] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise VideoInsightError("could not decode frames from video")
    return frames


def extract_stream_frames(
    stream_url: str,
    *,
    num_frames: int = 8,
    sample_interval_seconds: float = 0.75,
) -> list[np.ndarray]:
    """Sample consecutive RGB frames from a live RTSP stream."""

    capture = cv2.VideoCapture(stream_url)
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open camera stream")
        frames: list[np.ndarray] = []
        next_sample = time.monotonic()
        failures = 0
        deadline = next_sample + max(15.0, num_frames * sample_interval_seconds + 10.0)
        while len(frames) < num_frames and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 20:
                    break
                continue
            failures = 0
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.05))
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_sample = now + sample_interval_seconds
    finally:
        capture.release()
    if not frames:
        raise VideoInsightError("could not capture frames from camera stream")
    return frames


class VideoInsightService:
    """Lazily loads one quantized model and serializes GPU generation calls."""

    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "VIDEO_INSIGHT_MODEL_PATH", "/models/Qwen2.5-VL-3B-Instruct"
        )
        self.min_pixels = int(os.environ.get("VIDEO_INSIGHT_MIN_PIXELS", str(256 * 256)))
        self.max_pixels = int(os.environ.get("VIDEO_INSIGHT_MAX_PIXELS", str(512 * 512)))
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise VideoInsightError(
                "video insight dependencies are unavailable; install the video-insight extra"
            ) from exc

        if self._model is not None and self._processor is not None:
            return self._model, self._processor, torch
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor, torch
            model_path = Path(self.model_path)
            if not model_path.is_dir():
                raise VideoInsightError(
                    f"Qwen model is missing at {model_path}; run the deployment model downloader"
                )
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            load_options: dict[str, Any] = {"local_files_only": True}
            if torch.cuda.is_available():
                load_options.update(
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                    torch_dtype=torch.float16,
                )
            else:
                load_options.update(device_map="cpu", torch_dtype=torch.float32)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_options
            )
            model.eval()
            self._processor = processor
            self._model = model
        return self._model, self._processor, torch

    def interpret(
        self,
        frames: list[np.ndarray],
        query: str,
        *,
        max_new_tokens: int = 160,
    ) -> dict[str, object]:
        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append(
            {
                "type": "text",
                "text": (
                    "These are consecutive frames from a video, ordered from oldest to newest. "
                    "Analyze the sequence and answer the user's query briefly and precisely, "
                    "using the same language as the query.\n\n"
                    f"User query: {query.strip()}"
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=frames, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return {
            "text": answer,
            "frame_count": len(frames),
            "inference_seconds": round(elapsed, 3),
            "model": Path(self.model_path).name,
        }

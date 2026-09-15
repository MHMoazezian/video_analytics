"""Optimized Qwen2.5-VL inference for recorded videos and RTSP sequences."""

from __future__ import annotations

import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np


class VideoInsightError(RuntimeError):
    """A user-facing video interpretation failure."""


PERSIAN_TEXT = re.compile(r"[\u0600-\u06ff]")


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

    def preload(self) -> None:
        """Load model weights before the first dashboard request."""

        self._load()

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
        max_new_tokens: int = 80,
        detailed: bool = False,
    ) -> dict[str, object]:
        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        original_query = query.strip()
        translated_query = original_query
        translation_seconds = 0.0
        with self._inference_lock, torch.inference_mode():
            if PERSIAN_TEXT.search(original_query):
                translated_query, elapsed = self._generate(
                    model,
                    processor,
                    torch,
                    [{
                        "role": "user",
                        "content": (
                            "Translate the following Persian video-analysis question into natural "
                            "English. Preserve its exact intent. Output only the English translation.\n\n"
                            f"{original_query}"
                        ),
                    }],
                    max_new_tokens=64,
                )
                translation_seconds += elapsed

            answer_style = (
                "Give a detailed chronological analysis. Describe all relevant people, objects, "
                "actions, and changes across the frames; explain evidence related to the query, "
                "and clearly state uncertainty. Do not infer facts unsupported by the frames."
                if detailed
                else "Give a concise general answer focused on the user query."
            )
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({
                "type": "text",
                "text": (
                    "These are consecutive frames from a CCTV video, ordered from oldest to newest. "
                    "Analyze the sequence and answer the user query in English. Mention only details "
                    f"supported by the frames. {answer_style}\n\n"
                    f"User query: {translated_query}"
                ),
            })
            english_answer, inference_seconds = self._generate(
                model,
                processor,
                torch,
                [{"role": "user", "content": content}],
                images=frames,
                max_new_tokens=max_new_tokens,
            )
            persian_answer, elapsed = self._generate(
                model,
                processor,
                torch,
                [{
                    "role": "user",
                    "content": (
                        "Translate the following English video-analysis answer into fluent Persian. "
                        "Preserve all facts and uncertainty. Output only the Persian translation.\n\n"
                        f"{english_answer}"
                    ),
                }],
                max_new_tokens=min(512, max_new_tokens * 2),
            )
            translation_seconds += elapsed

        return {
            "text": persian_answer or english_answer,
            "english_text": english_answer,
            "translated_query": translated_query,
            "detailed": detailed,
            "frame_count": len(frames),
            "inference_seconds": round(inference_seconds, 3),
            "translation_seconds": round(translation_seconds, 3),
            "total_seconds": round(inference_seconds + translation_seconds, 3),
            "model": Path(self.model_path).name,
        }

    @staticmethod
    def _generate(
        model: Any,
        processor: Any,
        torch: Any,
        messages: list[dict[str, Any]],
        *,
        images: list[np.ndarray] | None = None,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_options: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }
        if images is not None:
            processor_options["images"] = images
        inputs = processor(**processor_options).to(next(model.parameters()).device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[:, inputs.input_ids.shape[1] :]
        text = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text, elapsed

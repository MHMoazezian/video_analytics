"""Optimized Qwen3-VL inference for recorded videos and RTSP sequences."""

from __future__ import annotations

import json
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


FIGHT_QUESTION = "Are there persons fighting in the video?"
FLOOR_CLEAN_QUESTION = "Is the floor of the scene clean?"

FRUIT_QUALITY_PROMPT = """Evaluate the visible freshness of the fruits in these images. The images may be one photo or ordered samples from one video. Judge only visible fruit and summarize the whole scene; when video frames repeat the same fruits, do not treat each appearance as a different fruit.

Use visible evidence such as natural color, discoloration, bruising, mold, soft or collapsed areas, wrinkling, dryness, and decay. Do not claim anything about taste, smell, internal quality, or food safety that cannot be seen. Ignore non-fruit objects. If no fruit is clearly visible, set has_fruit to false and use the label "نامشخص".

Return exactly one JSON object with no prose or markdown, using this schema:
{"has_fruit":true,"label":"تازه","freshness_score":95,"distribution":{"fresh":100,"middle":0,"rotten":0},"fruit_count_estimate":12,"confidence":90,"summary_fa":"توضیح کوتاه فارسی"}

Rules:
- freshness_score, confidence, and distribution values are integers from 0 to 100; distribution must total 100.
- fruit_count_estimate is a non-negative integer estimate, or null when it cannot be estimated reliably.
- summary_fa must be one concise Persian sentence grounded in visible evidence.
- label must be exactly one of: "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص".
- If essentially all visible fruits are fresh, use "تازه". If most are fresh but a minority show aging or defects, use "تقریباً تازه". Use "متوسط" for a mixed or mid-quality lot, "تقریباً فاسد" when most show substantial deterioration, and "فاسد" when essentially all show clear decay.
"""

FRUIT_QUALITY_LABELS = {
    "تازه",
    "تقریباً تازه",
    "متوسط",
    "تقریباً فاسد",
    "فاسد",
    "نامشخص",
}


def _parse_yes_no_answers(raw_output: str) -> dict[str, str]:
    """Convert the private model response into the public two-answer contract."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoInsightError("the video answers could not be determined") from exc

    def normalize(key: str) -> str:
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
            return value.strip().title()
        raise VideoInsightError("the video answers could not be determined")

    return {"fighting": normalize("fighting"), "floor_clean": normalize("floor_clean")}


def _parse_fruit_quality(raw_output: str) -> dict[str, object]:
    """Validate the model response before exposing it through the API."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoInsightError("the fruit quality could not be determined") from exc
    if not isinstance(payload, dict):
        raise VideoInsightError("the fruit quality could not be determined")

    has_fruit = payload.get("has_fruit")
    label = payload.get("label")
    summary = payload.get("summary_fa")
    distribution = payload.get("distribution")
    if not isinstance(has_fruit, bool) or label not in FRUIT_QUALITY_LABELS:
        raise VideoInsightError("the fruit quality could not be determined")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 500:
        raise VideoInsightError("the fruit quality could not be determined")
    if not isinstance(distribution, dict):
        raise VideoInsightError("the fruit quality could not be determined")

    def percentage(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VideoInsightError("the fruit quality could not be determined")
        rounded = round(value)
        if not 0 <= rounded <= 100:
            raise VideoInsightError("the fruit quality could not be determined")
        return rounded

    freshness_score = percentage(payload.get("freshness_score"))
    confidence = percentage(payload.get("confidence"))
    normalized_distribution = {
        key: percentage(distribution.get(key)) for key in ("fresh", "middle", "rotten")
    }
    if sum(normalized_distribution.values()) != 100:
        raise VideoInsightError("the fruit quality percentages must total 100")
    count = payload.get("fruit_count_estimate")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise VideoInsightError("the fruit quality could not be determined")
    if not has_fruit:
        label = "نامشخص"
        count = None

    return {
        "has_fruit": has_fruit,
        "label": label,
        "freshness_score": freshness_score,
        "distribution": normalized_distribution,
        "fruit_count_estimate": count,
        "confidence": confidence,
        "summary_fa": summary.strip(),
    }


def extract_image_frame(image_path: str | Path) -> np.ndarray:
    """Decode one still image as RGB for Qwen."""

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise VideoInsightError("could not decode image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_video_frames(
    video_path: str | Path,
    *,
    num_frames: int = 8,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> list[np.ndarray]:
    """Return evenly spaced RGB frames from the requested video-time window."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoInsightError("video does not contain any decodable frames")
        first_index = 0
        last_index = total_frames - 1
        if window_start_seconds is not None or window_end_seconds is not None:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise VideoInsightError("video frame rate is unavailable")
            first_index = max(0, int((window_start_seconds or 0.0) * fps))
            requested_end = window_end_seconds if window_end_seconds is not None else total_frames / fps
            last_index = min(total_frames - 1, max(first_index, int(requested_end * fps) - 1))
            if first_index >= total_frames:
                raise VideoInsightError("video time window is outside the recording")
        count = min(num_frames, last_index - first_index + 1)
        indexes = np.linspace(first_index, last_index, num=count, dtype=int)
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
            "VIDEO_INSIGHT_MODEL_PATH", "/models/Qwen3-VL-2B-Instruct"
        )
        self.min_pixels = int(os.environ.get("VIDEO_INSIGHT_MIN_PIXELS", str(256 * 256)))
        self.max_pixels = int(os.environ.get("VIDEO_INSIGHT_MAX_PIXELS", str(384 * 384)))
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
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
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
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_options
            )
            model.eval()
            self._processor = processor
            self._model = model
        return self._model, self._processor, torch

    def interpret(self, frames: list[np.ndarray]) -> dict[str, object]:
        """Return only normalized answers; never expose generated model text."""

        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        with self._inference_lock, torch.inference_mode():
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({
                "type": "text",
                "text": (
                    "These are consecutive frames from a CCTV video, ordered from oldest to newest. "
                    "Answer both questions using only visible evidence. Treat uncertainty as No. "
                    "Return exactly one JSON object with no prose or markdown. Each value must be "
                    'either "Yes" or "No". Example: '
                    '{"fighting":"No","floor_clean":"Yes"}.\n'
                    f"1. {FIGHT_QUESTION}\n2. {FLOOR_CLEAN_QUESTION}"
                ),
            })
            raw_output, inference_seconds = self._generate(
                model,
                processor,
                torch,
                [{"role": "user", "content": content}],
                images=frames,
                max_new_tokens=40,
            )

        return {
            "answers": _parse_yes_no_answers(raw_output),
            "frame_count": len(frames),
            "inference_seconds": round(inference_seconds, 3),
        }

    def interpret_fruit_quality(self, frames: list[np.ndarray]) -> dict[str, object]:
        """Score the visible freshness of a fruit image or sampled video."""

        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        with self._inference_lock, torch.inference_mode():
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({"type": "text", "text": FRUIT_QUALITY_PROMPT})
            raw_output, inference_seconds = self._generate(
                model,
                processor,
                torch,
                [{"role": "user", "content": content}],
                images=frames,
                max_new_tokens=220,
            )
        return {
            **_parse_fruit_quality(raw_output),
            "frame_count": len(frames),
            "inference_seconds": round(inference_seconds, 3),
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

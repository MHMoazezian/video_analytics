from __future__ import annotations

import re
import asyncio
import base64
from io import BytesIO
import json

import cv2
import numpy as np
import pytest
from contextlib import nullcontext
from fastapi import HTTPException, UploadFile

from app.insights.service import (
    FRUIT_ACTIONS_FA,
    FRUIT_DEFECT_LABELS_FA,
    FRUIT_FRAME_PROMPT,
    FRUIT_PRESENCE_PROMPT,
    FRUIT_QUALITY_DETAILED_PROMPT,
    FRUIT_QUALITY_PROMPT,
    NO_FRUIT_SUMMARY_FA,
    OUT_OF_MEMORY_MESSAGE,
    SampledFrame,
    VideoInsightError,
    VideoInsightService,
    _parse_fruit_details,
    _parse_fruit_quality,
    _parse_yes_no_answers,
    resolve_pixel_budget,
    resolve_precision,
    _repair_truncated_json,
    collapse_repetition,
    encode_thumbnail,
    extract_stream_frames,
    extract_video_frames,
    frame_statistics,
    fruit_grade,
    fruit_verdict_fa,
    sample_stream_frames,
    sample_video_frames,
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
    monkeypatch.setattr(service, "_fruit_is_visible", lambda *_a: (True, 0.0))
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


# --- detailed fruit-quality contract (analysis_version 2) -------------------

CORE_ANSWER = {
    "has_fruit": True, "label": "تقریباً تازه", "freshness_score": 78,
    "distribution": {"fresh": 75, "middle": 20, "rotten": 5},
    "fruit_count_estimate": 9, "confidence": 84, "summary_fa": "بیشتر میوه‌ها تازه هستند.",
}
DETAILED_ANSWER = {
    **CORE_ANSWER,
    "grade": "A",  # never trusted: the grade is derived from the score
    "fruit_types": [
        {"name_fa": "سیب", "share_percent": 70.4},
        {"name_fa": "", "share_percent": 30},
        {"name_fa": "موز", "share_percent": 140},
        "پرتقال",
    ],
    "defects": [
        {"type": "Bruising", "severity": "LOW", "affected_percent": 10, "note_fa": "کوفتگی جزئی"},
        {"type": "sunburn", "severity": "medium", "affected_percent": 5},
        {"type": "mold", "severity": "severe", "affected_percent": 5, "note_fa": "x"},
        {"type": "decay", "severity": "high", "affected_percent": "many", "note_fa": "x"},
        {"type": None, "severity": "low", "affected_percent": 1},
        ["mold"],
    ],
    "shelf_life_days_estimate": 4.6,
    "recommendation_fa": "  ابتدا این محموله فروخته شود.  ",
    "storage_advice_fa": 17,
}


def _service(
    monkeypatch, fake_torch: type | None = None, *, presence_check: bool = False
) -> VideoInsightService:
    """Service with a fake model. The one-word fruit-presence question is answered
    "yes" without a model call unless a test asks for the real check."""

    monkeypatch.setenv("VIDEO_INSIGHT_MODEL_PATH", "/srv/private/models/Qwen2.5-VL-3B-Instruct/")
    service = VideoInsightService()
    torch = fake_torch or type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), torch))
    if not presence_check:
        monkeypatch.setattr(service, "_fruit_is_visible", lambda *_a: (True, 0.0))
    return service


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((4, 6, 3), index, dtype=np.uint8) for index in range(count)]


def test_lenient_detail_parser_drops_bad_optional_entries() -> None:
    details = _parse_fruit_details(DETAILED_ANSWER)

    assert details["fruit_types"] == [{"name_fa": "سیب", "share_percent": 70}]
    assert details["defects"] == [
        {"type": "bruising", "label_fa": "کوفتگی", "severity": "low",
         "affected_percent": 10, "note_fa": "کوفتگی جزئی"},
        # An unlisted but well-formed defect is kept under "other".
        {"type": "other", "label_fa": "سایر", "severity": "medium",
         "affected_percent": 5, "note_fa": ""},
    ]
    assert details["shelf_life_days_estimate"] == 5
    # The model's advice is kept for the record only; the shown one follows the grade.
    assert details["model_recommendation_fa"] == "ابتدا این محموله فروخته شود."
    assert details["recommendation_fa"] is None
    assert details["storage_advice_fa"] is None


def test_lenient_detail_parser_never_raises_on_garbage() -> None:
    details = _parse_fruit_details({
        "fruit_types": "سیب", "defects": {"type": "mold"}, "shelf_life_days_estimate": True,
        "recommendation_fa": ["x"], "storage_advice_fa": "   ",
    })

    assert details == {
        "fruit_types": [], "defects": [], "shelf_life_days_estimate": None,
        "recommendation_fa": None, "model_recommendation_fa": None, "storage_advice_fa": None,
    }


def test_every_defect_type_has_a_persian_label() -> None:
    assert set(FRUIT_DEFECT_LABELS_FA) == {
        "bruising", "mold", "discoloration", "soft_spot", "wrinkling",
        "dryness", "decay", "cut_damage", "pest_damage", "other",
    }
    assert all(label.strip() for label in FRUIT_DEFECT_LABELS_FA.values())
    for defect_type in FRUIT_DEFECT_LABELS_FA:
        assert defect_type in FRUIT_QUALITY_DETAILED_PROMPT


@pytest.mark.parametrize(
    ("score", "grade"),
    [(100, "A"), (85, "A"), (84, "B"), (70, "B"), (69, "C"), (50, "C"), (49, "D"), (0, "D")],
)
def test_grade_is_derived_from_the_freshness_score(score: int, grade: str) -> None:
    assert fruit_grade(score, True) == grade


def test_grade_is_null_without_fruit() -> None:
    assert fruit_grade(95, False) is None
    assert fruit_grade(None, True) is None


def test_summary_response_carries_every_new_field_with_empty_defaults(monkeypatch) -> None:
    service = _service(monkeypatch)
    monkeypatch.setattr(service, "_generate", lambda *_a, **_k: (json.dumps(DETAILED_ANSWER), 1.5))

    result = service.interpret_fruit_quality(_frames(3))

    assert {key: result[key] for key in CORE_ANSWER} == CORE_ANSWER
    assert result["frame_count"] == 3
    assert result["inference_seconds"] == 1.5
    assert result["analysis_version"] == 2
    assert result["detail_level"] == "summary"
    assert result["model"] == "Qwen2.5-VL-3B-Instruct"
    assert result["grade"] == "B"
    # Detail fields are only filled for detail_level="detailed".
    assert result["fruit_types"] == [] and result["defects"] == []
    assert result["shelf_life_days_estimate"] is None
    # The recommended action follows the grade, so it exists at every detail level.
    assert result["recommendation_fa"] == FRUIT_ACTIONS_FA["B"]
    assert result["model_recommendation_fa"] is None and result["storage_advice_fa"] is None
    assert result["frames"] == []
    assert result["frame_statistics"] == {
        "analyzed": 0, "score_min": None, "score_max": None,
        "score_mean": None, "score_stddev": None,
    }
    assert result["total_seconds"] >= 0


def test_detailed_response_uses_the_detailed_prompt_and_server_side_grade(monkeypatch) -> None:
    service = _service(monkeypatch)
    calls: list[tuple[str, int]] = []

    def fake_generate(*args, **kwargs):
        calls.append((args[3][0]["content"][-1]["text"], kwargs["max_new_tokens"]))
        return json.dumps(DETAILED_ANSWER), 2.0

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(_frames(2), detail_level="detailed")

    assert calls == [(FRUIT_QUALITY_DETAILED_PROMPT, 700)]
    assert result["detail_level"] == "detailed"
    assert result["grade"] == "B"  # 78 -> B, although the model claimed "A"
    assert result["fruit_types"] == [{"name_fa": "سیب", "share_percent": 70}]
    assert [item["type"] for item in result["defects"]] == ["bruising", "other"]
    assert result["shelf_life_days_estimate"] == 5


def test_detailed_request_falls_back_to_the_summary_prompt_when_the_answer_is_unusable(monkeypatch) -> None:
    service = _service(monkeypatch)
    prompts: list[str] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3][0]["content"][-1]["text"])
        if len(prompts) == 1:
            return '{"has_fruit":true,"label":"تازه","freshness_score":90,"distri', 3.0
        return json.dumps(CORE_ANSWER), 1.0

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(_frames(2), detail_level="detailed")

    assert prompts == [FRUIT_QUALITY_DETAILED_PROMPT, FRUIT_QUALITY_PROMPT]
    assert result["label"] == "تقریباً تازه"
    assert result["detail_level"] == "detailed"
    assert result["defects"] == [] and result["fruit_types"] == []
    assert result["inference_seconds"] == 4.0


def test_detail_fields_are_empty_when_no_fruit_is_visible(monkeypatch) -> None:
    service = _service(monkeypatch)
    answer = {**DETAILED_ANSWER, "has_fruit": False}
    monkeypatch.setattr(service, "_generate", lambda *_a, **_k: (json.dumps(answer), 1.0))

    result = service.interpret_fruit_quality(_frames(1), detail_level="detailed")

    assert result["label"] == "نامشخص"
    assert result["grade"] is None
    assert result["defects"] == [] and result["fruit_types"] == []
    assert result["recommendation_fa"] is None


def test_per_frame_failure_is_isolated_to_that_frame(monkeypatch) -> None:
    service = _service(monkeypatch)
    frame_calls: list[int] = []

    def fake_generate(*args, **kwargs):
        prompt = args[3][0]["content"][-1]["text"]
        if prompt != FRUIT_FRAME_PROMPT:
            return json.dumps(CORE_ANSWER), 2.0
        assert len(kwargs["images"]) == 1
        frame_calls.append(int(kwargs["images"][0][0, 0, 0]))
        position = len(frame_calls)
        if position == 2:
            return "I cannot tell.", 0.2
        if position == 3:
            raise ValueError("tokenizer exploded at /srv/private/models")
        if position == 4:
            return '{"has_fruit":false,"label":"تازه","freshness_score":10,"note_fa":"میوه‌ای نیست"}', 0.2
        score = 90 if position == 1 else 80
        return json.dumps({"has_fruit": True, "label": "تازه", "freshness_score": score, "note_fa": "براق"}), 0.2

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(
        _frames(5), per_frame=True, timestamps=[0.0, 1.5, 3.0, 4.5, 6.0]
    )

    # The aggregate verdict is untouched by the two failing frames.
    assert result["label"] == "تقریباً تازه"
    assert frame_calls == [0, 1, 2, 3, 4]
    frames = result["frames"]
    assert [item["index"] for item in frames] == [0, 1, 2, 3, 4]
    assert [item["timestamp_seconds"] for item in frames] == [0.0, 1.5, 3.0, 4.5, 6.0]
    assert frames[0] == {
        "index": 0, "timestamp_seconds": 0.0, "has_fruit": True, "freshness_score": 90,
        "label": "تازه", "note_fa": "براق", "thumbnail": None, "error": None,
    }
    assert frames[1]["error"] == "the fruit quality could not be determined"
    assert frames[1]["freshness_score"] is None and frames[1]["has_fruit"] is None
    # Unexpected errors are reported generically: no internals leak to the client.
    assert frames[2]["error"] == "frame analysis failed"
    assert frames[2]["freshness_score"] is None and frames[2]["label"] is None
    assert frames[3]["has_fruit"] is False and frames[3]["label"] == "نامشخص"
    # Statistics: three analysed frames, scores only from the two that show fruit.
    assert result["frame_statistics"] == {
        "analyzed": 3, "score_min": 80, "score_max": 90, "score_mean": 85.0, "score_stddev": 5.0,
    }


def test_thumbnails_without_per_frame_analysis_leave_scores_null(monkeypatch) -> None:
    service = _service(monkeypatch)
    calls: list[str] = []

    def fake_generate(*args, **_kwargs):
        calls.append(args[3][0]["content"][-1]["text"])
        return json.dumps(CORE_ANSWER), 1.0

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(_frames(2), include_thumbnails=True)

    assert calls == [FRUIT_QUALITY_PROMPT]  # no per-frame model calls
    assert len(result["frames"]) == 2
    for item in result["frames"]:
        assert item["thumbnail"].startswith("data:image/jpeg;base64,")
        assert item["freshness_score"] is None and item["error"] is None
    assert result["frame_statistics"]["analyzed"] == 0


def _decode_data_url(data_url: str) -> np.ndarray:
    header, _, encoded = data_url.partition(",")
    assert header == "data:image/jpeg;base64"
    raw = base64.b64decode(encoded)
    assert raw[:2] == b"\xff\xd8"  # JPEG start-of-image marker
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_thumbnail_is_a_small_jpeg_with_correct_colours() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = 250  # pure red in RGB

    image = _decode_data_url(encode_thumbnail(frame))

    assert image.shape[:2] == (180, 320)
    blue, green, red = (int(value) for value in image[90, 160])
    assert red > 200 and blue < 60 and green < 60


def test_thumbnail_never_upscales_a_small_frame() -> None:
    image = _decode_data_url(encode_thumbnail(np.zeros((90, 160, 3), dtype=np.uint8)))

    assert image.shape[:2] == (90, 160)
    assert encode_thumbnail(np.zeros((0, 0, 3), dtype=np.uint8)) is None


class _OomTorch:
    """torch stand-in exposing only what the OOM handling touches."""

    class cuda:  # noqa: N801 - mirrors torch.cuda
        class OutOfMemoryError(RuntimeError):
            pass

        empty_cache_calls = 0

        @classmethod
        def empty_cache(cls) -> None:
            cls.empty_cache_calls += 1

    inference_mode = staticmethod(nullcontext)


def test_gpu_out_of_memory_is_retried_once_with_half_the_frames(monkeypatch) -> None:
    _OomTorch.cuda.empty_cache_calls = 0
    service = _service(monkeypatch, _OomTorch)
    attempts: list[int] = []

    def fake_generate(*_args, **kwargs):
        attempts.append(len(kwargs["images"]))
        if len(attempts) == 1:
            raise _OomTorch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.2 GiB")
        return json.dumps(CORE_ANSWER), 2.5

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(_frames(8))

    assert attempts == [8, 4]
    assert _OomTorch.cuda.empty_cache_calls == 1
    assert result["frame_count"] == 4
    assert result["label"] == "تقریباً تازه"


def test_gpu_out_of_memory_twice_becomes_a_clear_insight_error(monkeypatch) -> None:
    _OomTorch.cuda.empty_cache_calls = 0
    service = _service(monkeypatch, _OomTorch)
    attempts: list[int] = []

    def fake_generate(*_args, **kwargs):
        attempts.append(len(kwargs["images"]))
        # Older drivers surface OOM as a plain RuntimeError.
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(service, "_generate", fake_generate)
    with pytest.raises(VideoInsightError) as raised:
        service.interpret(_frames(3))

    assert attempts == [3, 1]
    assert str(raised.value) == OUT_OF_MEMORY_MESSAGE
    assert "memory" in OUT_OF_MEMORY_MESSAGE
    assert _OomTorch.cuda.empty_cache_calls == 2


class _FakeInputs(dict):
    """Processor output: unpackable into generate() and movable to a device."""

    def __init__(self, image_count: int) -> None:
        super().__init__(input_ids=np.zeros((1, 5), dtype=np.int64))
        self.input_ids = self["input_ids"]
        self.image_count = image_count

    def to(self, _device: object) -> "_FakeInputs":
        return self


class _FakeProcessor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.image_counts: list[int] = []

    def apply_chat_template(self, messages, **_options) -> str:
        return messages[0]["content"][-1]["text"]

    def __call__(self, *, text, images, return_tensors) -> _FakeInputs:
        self.image_counts.append(len(images))
        return _FakeInputs(len(images))

    def batch_decode(self, generated, **_options) -> list[str]:
        assert generated.shape == (1, 4)  # only the newly generated suffix is decoded
        return [self.answer]


class _FakeModel:
    """Runs out of GPU memory on the first generate() call only."""

    def __init__(self, torch: type) -> None:
        self.torch = torch
        self.calls = 0

    def parameters(self):
        yield type("Parameter", (), {"device": "cpu"})()

    def generate(self, *, input_ids, max_new_tokens, do_sample):
        self.calls += 1
        if self.calls == 1:
            raise self.torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 900 MiB")
        return np.zeros((1, input_ids.shape[1] + 4), dtype=np.int64)


def test_fake_model_that_runs_out_of_memory_once_is_retried_through_generate(monkeypatch) -> None:
    class Torch(_OomTorch):
        class cuda(_OomTorch.cuda):  # noqa: N801
            empty_cache_calls = 0

            @staticmethod
            def is_available() -> bool:
                return False

    model, processor = _FakeModel(Torch), _FakeProcessor(json.dumps(CORE_ANSWER))
    service = VideoInsightService()
    monkeypatch.setattr(service, "_load", lambda: (model, processor, Torch))
    monkeypatch.setattr(service, "_fruit_is_visible", lambda *_a: (True, 0.0))

    result = service.interpret_fruit_quality(_frames(6), include_thumbnails=True)

    assert model.calls == 2
    assert processor.image_counts == [6, 3]
    assert Torch.cuda.empty_cache_calls == 1
    assert result["frame_count"] == 3
    assert result["freshness_score"] == 78 and result["grade"] == "B"
    # Evidence still covers every sampled frame, not only the ones that fitted.
    assert len(result["frames"]) == 6


def test_other_runtime_errors_are_not_retried(monkeypatch) -> None:
    service = _service(monkeypatch, _OomTorch)
    attempts: list[int] = []

    def fake_generate(*_args, **kwargs):
        attempts.append(len(kwargs["images"]))
        raise RuntimeError("device-side assert triggered")

    monkeypatch.setattr(service, "_generate", fake_generate)
    with pytest.raises(RuntimeError, match="device-side assert"):
        service.interpret_fruit_quality(_frames(4))

    assert attempts == [4]


def test_video_insight_evidence_adds_middle_thumbnail_and_model(monkeypatch) -> None:
    service = _service(monkeypatch)
    monkeypatch.setattr(
        service, "_generate", lambda *_a, **_k: ('{"fighting":"No","floor_clean":"Yes"}', 1.0)
    )
    frames = [np.full((40, 60, 3), 10 * index, dtype=np.uint8) for index in range(5)]

    plain = service.interpret_video(frames)
    evidence = service.interpret_video(frames, include_thumbnail=True)

    assert plain["thumbnail"] is None
    assert plain["model"] == "Qwen2.5-VL-3B-Instruct"
    assert set(evidence) == {"answers", "frame_count", "inference_seconds", "thumbnail", "model"}
    middle = _decode_data_url(evidence["thumbnail"])
    assert middle.shape[:2] == (40, 60)
    assert abs(int(middle[20, 30, 0]) - 20) <= 3  # frames[2] is the middle frame


def test_sampled_video_frames_carry_their_video_time(monkeypatch) -> None:
    capture = _SeekableCapture(total=120, fps=10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    samples = sample_video_frames("sample.mp4", num_frames=4, window_start_seconds=2, window_end_seconds=5)

    assert all(isinstance(item, SampledFrame) for item in samples)
    assert [item.timestamp_seconds for item in samples] == [2.0, 2.9, 3.9, 4.9]
    assert [int(item.image[0, 0, 2]) for item in samples] == [20, 29, 39, 49]


def test_sampled_video_frames_without_a_frame_rate_have_no_time(monkeypatch) -> None:
    capture = _SeekableCapture(total=12, fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    samples = sample_video_frames("sample.mp4", num_frames=3)

    assert [item.timestamp_seconds for item in samples] == [None, None, None]


def test_sampled_stream_frames_are_timed_from_the_first_sample(monkeypatch) -> None:
    capture = _SeekableCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _url: capture)

    samples = sample_stream_frames("rtsp://mediamtx:8554/camera", num_frames=3, sample_interval_seconds=0)

    stamps = [item.timestamp_seconds for item in samples]
    assert stamps[0] == 0.0
    assert stamps == sorted(stamps)


def test_frame_statistics_ignores_failed_frames() -> None:
    assert frame_statistics([
        {"error": "frame analysis failed", "has_fruit": None, "freshness_score": None},
        {"error": None, "has_fruit": True, "freshness_score": 70},
    ]) == {"analyzed": 1, "score_min": 70, "score_max": 70, "score_mean": 70.0, "score_stddev": 0.0}


# --- HTTP layer ---------------------------------------------------------------


class _RecordingInsightService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def interpret_fruit_quality(self, frames, **options):
        if self.error is not None:
            raise self.error
        self.calls.append({"frames": len(frames), **options})
        return {"label": "تازه", "total_seconds": 0.0}

    def interpret_video(self, frames, **options):
        if self.error is not None:
            raise self.error
        self.calls.append({"frames": len(frames), **options})
        return {"answers": {"fighting": "No", "floor_clean": "Yes"}}


def _png_upload() -> UploadFile:
    ok, encoded = cv2.imencode(".png", np.full((12, 16, 3), 127, dtype=np.uint8))
    assert ok
    payload = encoded.tobytes()
    return UploadFile(file=BytesIO(payload), filename="fruit.png", size=len(payload))


def test_fruit_quality_route_forwards_the_new_form_fields(tmp_path, monkeypatch) -> None:
    from app.api import main

    service = _RecordingInsightService()
    monkeypatch.setattr(main, "video_insight_service", service)
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)

    response = asyncio.run(main.interpret_fruit_quality(
        media=_png_upload(), num_frames=8, detail_level="detailed",
        per_frame=True, include_thumbnails=True,
    ))

    assert service.calls == [{
        "frames": 1, "detail_level": "detailed", "per_frame": True,
        "include_thumbnails": True, "timestamps": [0.0],
    }]
    assert response["data"]["label"] == "تازه"
    assert response["data"]["total_seconds"] > 0
    assert list(tmp_path.iterdir()) == []  # the staged upload is gone


def test_stream_requests_default_to_todays_behaviour() -> None:
    from app.api import main

    fruit = main.StreamFruitQualityRequest(stream_url="rtsp://mediamtx:8554/camera")
    assert (fruit.detail_level, fruit.per_frame, fruit.include_thumbnails) == ("summary", False, False)
    assert main.StreamVideoInsightRequest(stream_url="rtsp://mediamtx:8554/camera").include_thumbnail is False
    with pytest.raises(ValueError):
        main.StreamFruitQualityRequest(stream_url="rtsp://mediamtx:8554/camera", detail_level="full")
    with pytest.raises(ValueError):
        main.StreamFruitQualityRequest(stream_url="rtsp://mediamtx:8554/camera", include_thumbnail=True)


def test_stream_routes_forward_the_new_json_fields(monkeypatch) -> None:
    from app.api import main
    import app.insights.service as insight_module

    service = _RecordingInsightService()
    monkeypatch.setattr(main, "video_insight_service", service)
    samples = [SampledFrame(np.zeros((4, 6, 3), dtype=np.uint8), stamp) for stamp in (0.0, 1.4)]
    monkeypatch.setattr(insight_module, "sample_stream_frames", lambda *_a, **_k: samples)
    monkeypatch.setattr(insight_module, "extract_stream_frames", lambda *_a, **_k: [item.image for item in samples])

    asyncio.run(main.interpret_live_fruit_quality(main.StreamFruitQualityRequest(
        stream_url="rtsp://mediamtx:8554/camera", num_frames=2, per_frame=True,
    )))
    asyncio.run(main.interpret_live_video(main.StreamVideoInsightRequest(
        stream_url="rtsp://mediamtx:8554/camera", num_frames=2, include_thumbnail=True,
    )))

    assert service.calls == [
        {"frames": 2, "detail_level": "summary", "per_frame": True,
         "include_thumbnails": False, "timestamps": [0.0, 1.4]},
        {"frames": 2, "include_thumbnail": True},
    ]


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (VideoInsightError(OUT_OF_MEMORY_MESSAGE), OUT_OF_MEMORY_MESSAGE),
        (RuntimeError("CUDA error: an illegal memory access at /srv/private"), None),
    ],
)
def test_insight_routes_answer_503_for_every_failure(tmp_path, monkeypatch, error, detail) -> None:
    from app.api import main

    monkeypatch.setattr(main, "video_insight_service", _RecordingInsightService(error))
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(main.interpret_fruit_quality(
            media=_png_upload(), num_frames=8, detail_level="summary",
            per_frame=False, include_thumbnails=False,
        ))

    assert raised.value.status_code == 503
    assert raised.value.detail == (detail or main.INSIGHT_UNAVAILABLE_DETAIL)
    assert "/srv/private" not in str(raised.value.detail)


def test_insight_error_mapping_keeps_client_errors() -> None:
    from app.api import main

    with pytest.raises(HTTPException) as raised:
        with main._insight_errors():
            raise HTTPException(status_code=413, detail="uploaded file is too large")

    assert raised.value.status_code == 413


def _post_multipart(path: str, fields: dict[str, str], file_field: str, filename: str, payload: bytes):
    """POST a real multipart body through the ASGI stack (httpx is not installed)."""

    from app.api import main

    boundary = "----tarebar-test-boundary"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
        + payload + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
        "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    messages: list[dict[str, object]] = []
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            await asyncio.sleep(3600)  # the client stays connected
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    asyncio.run(asyncio.wait_for(main.app(scope, receive, send), timeout=30))
    raw = b"".join(item.get("body", b"") for item in messages[1:])  # type: ignore[misc]
    return messages[0]["status"], json.loads(raw)


def test_fruit_quality_multipart_form_fields_are_optional_and_validated(tmp_path, monkeypatch) -> None:
    from app.api import main

    service = _RecordingInsightService()
    monkeypatch.delenv("SERVICE_AUTH_SECRET", raising=False)
    monkeypatch.setattr(main, "video_insight_service", service)
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    ok, encoded = cv2.imencode(".png", np.full((12, 16, 3), 127, dtype=np.uint8))
    assert ok
    image = encoded.tobytes()

    status, _body = _post_multipart("/api/v1/fruit-quality", {}, "media", "fruit.png", image)
    assert status == 200
    status, _body = _post_multipart(
        "/api/v1/fruit-quality",
        {"detail_level": "detailed", "per_frame": "true", "include_thumbnails": "true"},
        "media", "fruit.png", image,
    )
    assert status == 200
    status, body = _post_multipart(
        "/api/v1/fruit-quality", {"detail_level": "everything"}, "media", "fruit.png", image
    )
    assert status == 422 and body["detail"][0]["loc"] == ["body", "detail_level"]

    assert [(call["detail_level"], call["per_frame"], call["include_thumbnails"]) for call in service.calls] == [
        ("summary", False, False),  # omitted fields keep today's behaviour
        ("detailed", True, True),
    ]


def test_fruit_prompts_contain_no_example_values_a_small_model_could_copy():
    """The 3B model copied the numbers of an example JSON verbatim (78, 75/20/5, 90)."""
    from app.insights import service

    for prompt in (
        service.FRUIT_QUALITY_PROMPT,
        service.FRUIT_QUALITY_DETAILED_PROMPT,
        service.FRUIT_FRAME_PROMPT,
    ):
        # No key may be followed by a literal value (number, true/false or quoted text).
        assert not re.search(r'"(?:freshness_score|confidence|fresh|middle|rotten|share_percent|affected_percent)"\s*:\s*\d', prompt)
        assert '{"has_fruit"' not in prompt.replace(" ", "")
        assert "Scoring guide" in prompt
        for label in service.FRUIT_QUALITY_LABELS:
            assert label in prompt


def test_echoed_instruction_prefixes_are_removed_from_model_text():
    from app.insights.service import _optional_text, clean_model_text

    assert clean_model_text("توضیح کوتاه فارسی: مخلوطی از سیب و پرتقال.") == "مخلوطی از سیب و پرتقال."
    assert clean_model_text("توصیه نگهداری کوتاه فارسی: در جای خنک نگهداری شود.") == "در جای خنک نگهداری شود."
    assert clean_model_text("سیب‌ها براق و بدون لک هستند.") == "سیب‌ها براق و بدون لک هستند."
    assert _optional_text("توصیه کوتاه فارسی:  ") is None


def test_no_fruit_answer_is_a_result_not_an_error() -> None:
    result = _parse_fruit_quality(
        '{"has_fruit":false,"label":null,"freshness_score":null,"distribution":null,'
        '"fruit_count_estimate":null,"confidence":92,"summary_fa":null}'
    )

    assert result["has_fruit"] is False
    assert result["label"] == "نامشخص"
    assert result["freshness_score"] == 0
    assert result["distribution"] == {"fresh": 0, "middle": 0, "rotten": 0}
    assert result["fruit_count_estimate"] is None
    assert result["confidence"] == 92
    assert result["summary_fa"]


def test_distribution_close_to_one_hundred_is_rebalanced() -> None:
    result = _parse_fruit_quality(
        '{"has_fruit":true,"label":"متوسط","freshness_score":55,'
        '"distribution":{"fresh":60,"middle":30,"rotten":15},'
        '"fruit_count_estimate":8,"confidence":70,"summary_fa":"کیفیت متوسط است."}'
    )

    assert sum(result["distribution"].values()) == 100
    assert result["distribution"]["fresh"] > result["distribution"]["middle"] > result["distribution"]["rotten"]


def test_repeated_fruit_names_are_merged() -> None:
    from app.insights.service import _parse_fruit_details

    details = _parse_fruit_details({"fruit_types": [
        {"name_fa": "پرتقال", "share_percent": 50},
        {"name_fa": "لیمو", "share_percent": 20},
        {"name_fa": "پرتقال", "share_percent": 20},
    ]})

    assert details["fruit_types"] == [
        {"name_fa": "پرتقال", "share_percent": 70},
        {"name_fa": "لیمو", "share_percent": 20},
    ]


# --- presence check, cut-off answers and repetition loops -------------------


def test_scene_without_fruit_stops_after_the_one_word_presence_question(monkeypatch) -> None:
    service = _service(monkeypatch, presence_check=True)
    calls: list[tuple[str, int, int]] = []

    def fake_generate(*args, **kwargs):
        calls.append((args[3][0]["content"][-1]["text"], len(kwargs["images"]), kwargs["max_new_tokens"]))
        return "No.", 0.4

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(
        _frames(8), detail_level="detailed", per_frame=True, include_thumbnails=True
    )

    # One short question on three evenly spaced frames; no scoring, no per-frame calls.
    assert calls == [(FRUIT_PRESENCE_PROMPT, 3, 4)]
    assert result["has_fruit"] is False and result["label"] == "نامشخص"
    assert result["freshness_score"] == 0 and result["grade"] is None
    assert result["summary_fa"] == NO_FRUIT_SUMMARY_FA
    assert result["defects"] == [] and result["recommendation_fa"] is None
    assert result["frame_count"] == 8 and result["inference_seconds"] == 0.4
    # Evidence thumbnails are still returned so the user sees what was judged.
    assert len(result["frames"]) == 8
    assert all(item["thumbnail"] and item["freshness_score"] is None for item in result["frames"])


@pytest.mark.parametrize("presence_answer", ["Yes", "yes, several apples", "maybe", ""])
def test_anything_but_a_clear_no_continues_to_the_scoring_prompt(monkeypatch, presence_answer) -> None:
    service = _service(monkeypatch, presence_check=True)
    prompts: list[str] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3][0]["content"][-1]["text"])
        if prompts[-1] == FRUIT_PRESENCE_PROMPT:
            return presence_answer, 0.5
        return json.dumps(CORE_ANSWER), 2.0

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality(_frames(2))

    assert prompts == [FRUIT_PRESENCE_PROMPT, FRUIT_QUALITY_PROMPT]
    assert result["has_fruit"] is True and result["freshness_score"] == 78
    assert result["inference_seconds"] == 2.5


RUNAWAY_ANSWER = (
    '```json\n{ "has_fruit": true, "label": "تقریباً تازه", "freshness_score": 75, '
    '"distribution": {"fresh": 60, "middle": 30, "rotten": 10}, "fruit_count_estimate": 6, '
    '"confidence": 80, "summary_fa": "چند سیب و پرتقال روی میز دیده می‌شود. برخی میوه‌ها براق هستند، اما برخی دیگر نرم'
    + " و بعضی میوه‌ها نرم و نرمتر" * 12
)


def test_answer_cut_off_inside_a_repetition_loop_is_recovered() -> None:
    result = _parse_fruit_quality(RUNAWAY_ANSWER)

    assert result["freshness_score"] == 75
    assert result["distribution"] == {"fresh": 60, "middle": 30, "rotten": 10}
    assert result["fruit_count_estimate"] == 6
    # The loop is dropped; the complete sentence in front of it is kept.
    assert result["summary_fa"] == "چند سیب و پرتقال روی میز دیده می‌شود."


def test_answer_cut_off_before_the_core_fields_is_still_an_error() -> None:
    with pytest.raises(VideoInsightError):
        _parse_fruit_quality('{"has_fruit": true, "label": "تازه", "freshness_sc')
    with pytest.raises(VideoInsightError):
        _parse_fruit_quality("I cannot answer that")


def test_truncated_detail_answer_keeps_the_complete_members() -> None:
    cut = (
        '{"has_fruit": true, "label": "تازه", "freshness_score": 91, '
        '"distribution": {"fresh": 95, "middle": 5, "rotten": 0}, "fruit_count_estimate": null, '
        '"confidence": 88, "summary_fa": "سیب‌های براق و سالم.", '
        '"fruit_types": [{"name_fa": "سیب", "share_percent": 100}], '
        '"defects": [{"type": "bruising", "severity": "low", "affected_percent": 5, "note_fa": "روی دو سیب"}, {"type": "mo'
    )
    payload = _repair_truncated_json(cut)

    assert payload is not None
    assert payload["freshness_score"] == 91
    assert payload["fruit_types"] == [{"name_fa": "سیب", "share_percent": 100}]
    assert [item["type"] for item in payload["defects"]] == ["bruising"]
    assert _repair_truncated_json('{"a": 1}') is None  # complete objects are not "repaired"
    assert _repair_truncated_json('{"a": [1, 2}') is None  # mismatched brackets are not guessed at


def test_collapse_repetition_only_touches_real_loops() -> None:
    normal = "سیب‌ها تازه و براق هستند و پرتقال‌ها کمی چروکیده‌اند."
    assert collapse_repetition(normal) == normal
    # Saying a word twice is ordinary language, not a loop.
    assert collapse_repetition("خیلی خیلی تازه است.") == "خیلی خیلی تازه است."
    looped = "میوه‌ها نرم" + " و نرمتر" * 9
    assert collapse_repetition(looped) == "میوه‌ها نرم و نرمتر…"


def test_overlong_summary_is_shortened_instead_of_rejected() -> None:
    answer = {**CORE_ANSWER, "summary_fa": "الف " * 400}
    result = _parse_fruit_quality(json.dumps(answer))

    assert len(result["summary_fa"]) <= 500


def test_cut_off_text_ends_at_the_last_complete_sentence() -> None:
    cut = json.dumps({**CORE_ANSWER, "summary_fa": "X"}, ensure_ascii=False).replace(
        '"X"}', '"سیب‌ها چروکیده و قهوه‌ای شده‌اند. روی پوست لکه‌های نرم دیده می‌شود و بدون دیدن'
    )
    result = _parse_fruit_quality(cut)

    assert result["summary_fa"] == "سیب‌ها چروکیده و قهوه‌ای شده‌اند."


def test_first_person_chatter_is_not_shown_as_the_summary() -> None:
    chatter = "برای این تصویر، می‌توانم کلمات زبان فارسی را ارائه دهم. این تصویر یک سیب خشک و چروکیده نشان می‌دهد."
    result = _parse_fruit_quality(json.dumps({**CORE_ANSWER, "summary_fa": chatter}))
    assert result["summary_fa"] == "این تصویر یک سیب خشک و چروکیده نشان می‌دهد."

    only_chatter = "هر چه بهتر می‌توانم تصویر را مشاهده کنم، بهتر خواهم شناخت که میوه‌ها چگونه هستند."
    result = _parse_fruit_quality(json.dumps({**CORE_ANSWER, "summary_fa": only_chatter}))
    # Nothing usable is left, so the validated numbers are put into words instead.
    assert result["summary_fa"] == fruit_verdict_fa(result)


def test_verdict_is_composed_from_the_validated_numbers(monkeypatch) -> None:
    verdict = fruit_verdict_fa(CORE_ANSWER)

    assert verdict == (
        "کیفیت ظاهری «تقریباً تازه» ارزیابی شد: امتیاز تازگی ۷۸ از ۱۰۰ (درجه دو). "
        "حدود ۷۵٪ میوه‌ها تازه، ۲۰٪ متوسط و ۵٪ فاسد به نظر می‌رسند. "
        "تعداد تقریبی میوه‌های قابل مشاهده: ۹."
    )
    assert fruit_verdict_fa({**CORE_ANSWER, "has_fruit": False}) == NO_FRUIT_SUMMARY_FA

    service = _service(monkeypatch)
    answer = {key: value for key, value in CORE_ANSWER.items() if key != "summary_fa"}
    monkeypatch.setattr(service, "_generate", lambda *_a, **_k: (json.dumps(answer), 1.0))
    result = service.interpret_fruit_quality(_frames(1))
    # A missing sentence is no longer an error: the verdict stands in for it.
    assert result["verdict_fa"] == verdict and result["summary_fa"] == verdict


@pytest.mark.parametrize(("score", "grade"), [(96, "A"), (78, "B"), (55, "C"), (30, "D")])
def test_recommended_action_follows_the_grade_not_the_model(monkeypatch, score: int, grade: str) -> None:
    service = _service(monkeypatch)
    answer = {**DETAILED_ANSWER, "freshness_score": score, "recommendation_fa": "این میوه را سریع بفروشید."}
    monkeypatch.setattr(service, "_generate", lambda *_a, **_k: (json.dumps(answer), 1.0))

    result = service.interpret_fruit_quality(_frames(1), detail_level="detailed")

    assert result["recommendation_fa"] == FRUIT_ACTIONS_FA[grade]
    assert result["model_recommendation_fa"] == "این میوه را سریع بفروشید."


def test_a_sentence_in_place_of_a_fruit_name_is_dropped() -> None:
    details = _parse_fruit_details({
        "fruit_types": [
            {"name_fa": "فروختن این میوه بازدار است.", "share_percent": 100},
            {"name_fa": "سیب زرد", "share_percent": 60},
            {"name_fa": "میوه‌ای که در تصویر دیده می‌شود", "share_percent": 40},
        ]
    })

    assert details["fruit_types"] == [{"name_fa": "سیب زرد", "share_percent": 60}]


# --- model precision and image resolution follow the GPU ---------------------

GIB = 1024**3
WEIGHTS_3B = int(7.1 * GIB)
WEIGHTS_7B = int(16.6 * GIB)


@pytest.mark.parametrize(
    ("weights", "free_gib", "expected"),
    [
        (WEIGHTS_3B, 3.2, "4bit"),   # the 4 GB development laptop
        (WEIGHTS_3B, 11.0, "4bit"),  # 7.1 x 1.2 + 4 GiB headroom does not fit in 11
        (WEIGHTS_3B, 15.0, "bf16"),  # a 16 GB card runs the 3B model unquantized
        (WEIGHTS_7B, 22.0, "4bit"),
        (WEIGHTS_7B, 40.0, "bf16"),  # a 48 GB card runs the 7B model unquantized
    ],
)
def test_auto_precision_never_quantizes_a_model_the_gpu_can_hold(weights, free_gib, expected) -> None:
    assert resolve_precision(
        "auto", weights_bytes=weights, free_bytes=int(free_gib * GIB), bf16_supported=True
    ) == expected


def test_explicit_precision_is_honoured_and_bf16_degrades_to_fp16_on_old_gpus() -> None:
    small = {"weights_bytes": WEIGHTS_7B, "free_bytes": 2 * GIB}
    assert resolve_precision("bf16", bf16_supported=True, **small) == "bf16"
    assert resolve_precision("8bit", bf16_supported=True, **small) == "8bit"
    assert resolve_precision("4bit", bf16_supported=True, weights_bytes=1, free_bytes=80 * GIB) == "4bit"
    assert resolve_precision("bf16", bf16_supported=False, **small) == "fp16"
    assert resolve_precision(
        "auto", weights_bytes=WEIGHTS_3B, free_bytes=40 * GIB, bf16_supported=False
    ) == "fp16"


def test_pixel_budget_keeps_a_camera_frame_at_full_resolution_on_a_large_gpu() -> None:
    small = resolve_pixel_budget(None, None, total_bytes=4 * GIB)
    large = resolve_pixel_budget(None, None, total_bytes=24 * GIB)

    assert small == (256 * 256, 512 * 512)
    assert large == (256 * 28 * 28, 1280 * 28 * 28)
    assert large[1] >= 1280 * 720  # a 720p frame is not downscaled
    # Explicit settings always win, and the minimum never exceeds the maximum.
    assert resolve_pixel_budget(None, 2_000_000, total_bytes=4 * GIB) == (256 * 256, 2_000_000)
    assert resolve_pixel_budget(900_000, 400_000, total_bytes=24 * GIB) == (400_000, 400_000)


def test_precision_and_pixel_settings_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_INSIGHT_PRECISION", " BF16 ")
    monkeypatch.setenv("VIDEO_INSIGHT_MAX_PIXELS", "1003520")
    monkeypatch.setenv("VIDEO_INSIGHT_MIN_PIXELS", "")  # an empty compose default means "unset"
    service = VideoInsightService()

    assert service.describe() == {
        "model": "Qwen2.5-VL-3B-Instruct", "loaded": False, "requested_precision": "bf16",
        "precision": None, "min_pixels": None, "max_pixels": 1003520,
    }

    monkeypatch.setenv("VIDEO_INSIGHT_PRECISION", "int3")
    with pytest.raises(ValueError, match="VIDEO_INSIGHT_PRECISION"):
        VideoInsightService()
    monkeypatch.setenv("VIDEO_INSIGHT_PRECISION", "auto")
    monkeypatch.setenv("VIDEO_INSIGHT_MAX_PIXELS", "-5")
    with pytest.raises(ValueError, match="VIDEO_INSIGHT_MAX_PIXELS"):
        VideoInsightService()


def _fake_model_stack(monkeypatch, tmp_path, *, free_gib: float, total_gib: float, weights_gib: float):
    """Stand-ins for torch and transformers that record how the model is loaded."""

    import sys
    import types

    model_dir = tmp_path / "Qwen2.5-VL-7B-Instruct"
    model_dir.mkdir()
    with open(model_dir / "model-00001-of-00001.safetensors", "wb") as handle:
        handle.truncate(int(weights_gib * GIB))  # sparse file: size without disk use
    recorded: dict[str, object] = {}

    class _Cuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        mem_get_info = staticmethod(lambda _index: (int(free_gib * GIB), int(total_gib * GIB)))
        is_bf16_supported = staticmethod(lambda: True)
        get_device_properties = staticmethod(
            lambda _index: types.SimpleNamespace(total_memory=int(total_gib * GIB))
        )

    torch = types.SimpleNamespace(cuda=_Cuda, bfloat16="bf16-dtype", float16="fp16-dtype", float32="fp32-dtype")

    class _Processor:
        @staticmethod
        def from_pretrained(_path, **options):
            recorded["processor"] = options
            return object()

    class _Model:
        @staticmethod
        def from_pretrained(_path, **options):
            recorded["model"] = options
            return types.SimpleNamespace(eval=lambda: None)

    transformers = types.SimpleNamespace(
        AutoProcessor=_Processor,
        Qwen2_5_VLForConditionalGeneration=_Model,
        BitsAndBytesConfig=lambda **options: {"bitsandbytes": options},
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("VIDEO_INSIGHT_MODEL_PATH", str(model_dir))
    monkeypatch.delenv("VIDEO_INSIGHT_PRECISION", raising=False)
    monkeypatch.delenv("VIDEO_INSIGHT_MIN_PIXELS", raising=False)
    monkeypatch.delenv("VIDEO_INSIGHT_MAX_PIXELS", raising=False)
    return recorded


def test_large_gpu_loads_the_model_unquantized_at_full_image_resolution(monkeypatch, tmp_path) -> None:
    recorded = _fake_model_stack(monkeypatch, tmp_path, free_gib=44, total_gib=48, weights_gib=16.6)
    service = VideoInsightService()
    service.preload()

    assert "quantization_config" not in recorded["model"]
    assert recorded["model"]["torch_dtype"] == "bf16-dtype"
    assert recorded["model"]["device_map"] == "auto"
    assert recorded["processor"]["max_pixels"] == 1280 * 28 * 28
    assert service.describe() == {
        "model": "Qwen2.5-VL-7B-Instruct", "loaded": True, "requested_precision": "auto",
        "precision": "bf16", "min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28,
    }


def test_small_gpu_keeps_todays_four_bit_profile(monkeypatch, tmp_path) -> None:
    recorded = _fake_model_stack(monkeypatch, tmp_path, free_gib=3.2, total_gib=4, weights_gib=7.1)
    service = VideoInsightService()
    service.preload()

    assert recorded["model"]["quantization_config"] == {
        "bitsandbytes": {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "fp16-dtype",
        }
    }
    assert recorded["model"]["torch_dtype"] == "fp16-dtype"
    assert (recorded["processor"]["min_pixels"], recorded["processor"]["max_pixels"]) == (256 * 256, 512 * 512)
    assert service.describe()["precision"] == "4bit"

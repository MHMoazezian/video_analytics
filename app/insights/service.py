"""Optimized Qwen2.5-VL inference for recorded videos and RTSP sequences."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gc
import json
import logging
import os
from pathlib import Path
import re
import statistics
from threading import Lock
import time
from typing import Any, Sequence

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class VideoInsightError(RuntimeError):
    """A user-facing video interpretation failure."""


ANALYSIS_VERSION = 2
DETAIL_LEVELS = ("summary", "detailed")
THUMBNAIL_MAX_WIDTH = 320
THUMBNAIL_JPEG_QUALITY = 70
OUT_OF_MEMORY_MESSAGE = (
    "the GPU ran out of memory while analysing the frames; "
    "retry with fewer frames or when the GPU is less busy"
)


FIGHT_QUESTION = "Are there persons fighting in the video?"
FLOOR_CLEAN_QUESTION = "Is the floor of the scene clean?"

FRUIT_QUALITY_PROMPT = """Evaluate the visible freshness of the fruits in these images. The images may be one photo or ordered samples from one video. Judge only visible fruit and summarize the whole scene; when video frames repeat the same fruits, do not treat each appearance as a different fruit.

Use visible evidence such as natural color, discoloration, bruising, mold, soft or collapsed areas, wrinkling, dryness, and decay. Do not claim anything about taste, smell, internal quality, or food safety that cannot be seen. Ignore non-fruit objects. If no fruit is clearly visible, set has_fruit to false and use the label "نامشخص".

Scoring guide for freshness_score (judge this image, never reuse a number from these instructions):
- 90 to 100: firm, glossy, evenly coloured fruit with no visible defect.
- 70 to 89: fresh overall, with slight dullness or a few small blemishes.
- 50 to 69: noticeable bruising, wrinkling, soft areas or discoloration on many fruits.
- 25 to 49: widespread soft spots, mold or decay.
- 0 to 24: almost everything is rotten.

Return exactly one JSON object and nothing else: no prose, no markdown, no code fence. The object has exactly these keys:
- "has_fruit": true or false.
- "label": exactly one of "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص". Use "تازه" when essentially all visible fruits are fresh, "تقریباً تازه" when most are fresh but a minority show aging or defects, "متوسط" for a mixed or mid-quality lot, "تقریباً فاسد" when most show substantial deterioration, and "فاسد" when essentially all show clear decay.
- "freshness_score": integer from 0 to 100 following the scoring guide.
- "distribution": an object with the integer keys "fresh", "middle" and "rotten", the percentage of visible fruit in each state. The three values must total 100.
- "fruit_count_estimate": integer count of the individual fruits you can see, or null when it cannot be estimated reliably.
- "confidence": integer from 0 to 100, how sure you are given image sharpness, lighting and how much of the fruit is visible.
- "summary_fa": one concise sentence written in Persian that names the fruit you see and the visible evidence behind your score.

Write every Persian text value as a real sentence about this image. Never copy wording from these instructions into a value.
"""

FRUIT_PRESENCE_PROMPT = (
    "Look carefully at the image or images. Is at least one real fruit or vegetable "
    "clearly visible? Answer with exactly one word: Yes or No."
)
# The presence check only needs a glimpse of the scene, not every sampled frame.
FRUIT_PRESENCE_MAX_FRAMES = 3

FRUIT_QUALITY_LABELS = {
    "تازه",
    "تقریباً تازه",
    "متوسط",
    "تقریباً فاسد",
    "فاسد",
    "نامشخص",
}

FRUIT_DEFECT_LABELS_FA = {
    "bruising": "کوفتگی",
    "mold": "کپک‌زدگی",
    "discoloration": "تغییر رنگ",
    "soft_spot": "لکه نرم",
    "wrinkling": "چروکیدگی",
    "dryness": "خشکی",
    "decay": "پوسیدگی",
    "cut_damage": "بریدگی",
    "pest_damage": "آفت‌زدگی",
    "other": "سایر",
}
FRUIT_DEFECT_SEVERITIES = ("low", "medium", "high")
MAX_FRUIT_TYPES = 10
MAX_FRUIT_DEFECTS = 12

FRUIT_QUALITY_DETAILED_PROMPT = """Evaluate the visible freshness of the fruits in these images. The images may be one photo or ordered samples from one video. Judge only visible fruit and summarize the whole scene; when video frames repeat the same fruits, do not treat each appearance as a different fruit.

Use visible evidence such as natural color, discoloration, bruising, mold, soft or collapsed areas, wrinkling, dryness, and decay. Do not claim anything about taste, smell, internal quality, or food safety that cannot be seen. Ignore non-fruit objects. If no fruit is clearly visible, set has_fruit to false and use the label "نامشخص".

Scoring guide for freshness_score (judge this image, never reuse a number from these instructions):
- 90 to 100: firm, glossy, evenly coloured fruit with no visible defect.
- 70 to 89: fresh overall, with slight dullness or a few small blemishes.
- 50 to 69: noticeable bruising, wrinkling, soft areas or discoloration on many fruits.
- 25 to 49: widespread soft spots, mold or decay.
- 0 to 24: almost everything is rotten.

Return exactly one JSON object and nothing else: no prose, no markdown, no code fence. The object has exactly these keys:
- "has_fruit": true or false.
- "label": exactly one of "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص". Use "تازه" when essentially all visible fruits are fresh, "تقریباً تازه" when most are fresh but a minority show aging or defects, "متوسط" for a mixed or mid-quality lot, "تقریباً فاسد" when most show substantial deterioration, and "فاسد" when essentially all show clear decay.
- "freshness_score": integer from 0 to 100 following the scoring guide.
- "distribution": an object with the integer keys "fresh", "middle" and "rotten", the percentage of visible fruit in each state. The three values must total 100.
- "fruit_count_estimate": integer count of the individual fruits you can see, or null when it cannot be estimated reliably.
- "confidence": integer from 0 to 100, how sure you are given image sharpness, lighting and how much of the fruit is visible.
- "summary_fa": one concise sentence written in Persian that names the fruit you see and the visible evidence behind your score.
- "fruit_types": a list with one object per visible kind of fruit, each with "name_fa" (the Persian name) and "share_percent" (integer share, 0 to 100, of the visible fruit).
- "defects": a list with one object per defect that is actually visible, or an empty list when none is visible. Each object has "type" (exactly one of: bruising, mold, discoloration, soft_spot, wrinkling, dryness, decay, cut_damage, pest_damage, other), "severity" (exactly one of: low, medium, high), "affected_percent" (integer share, 0 to 100, of the visible fruit showing that defect) and "note_fa" (one short Persian phrase saying where or how it shows).
- "shelf_life_days_estimate": integer estimate of the remaining days at room conditions judged only from the visible condition, or null when it cannot be estimated.
- "recommendation_fa": one concise Persian sentence telling the seller what to do with this lot, for example sell it first, discount it, sort out the damaged fruit, or discard it.
- "storage_advice_fa": one concise Persian sentence about how to store this lot.

Write every Persian text value as a real sentence about this image. Never copy wording from these instructions into a value.
"""

FRUIT_FRAME_PROMPT = """Evaluate the visible freshness of the fruit in this single image. Judge only visible evidence such as color, bruising, mold, soft areas, wrinkling, dryness, and decay. Ignore non-fruit objects. If no fruit is clearly visible, set has_fruit to false and use the label "نامشخص".

Scoring guide for freshness_score (judge this image, never reuse a number from these instructions): 90 to 100 no visible defect; 70 to 89 a few small blemishes; 50 to 69 noticeable bruising, wrinkling or discoloration; 25 to 49 widespread soft spots, mold or decay; 0 to 24 almost everything is rotten.

Return exactly one JSON object and nothing else: no prose, no markdown, no code fence. The object has exactly these keys:
- "has_fruit": true or false.
- "label": exactly one of "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص".
- "freshness_score": integer from 0 to 100 following the scoring guide.
- "note_fa": one short Persian phrase naming the visible evidence. Never copy wording from these instructions.
"""


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


def _fruit_json_object(raw_output: str) -> dict[str, object]:
    """Extract the single JSON object the fruit prompts ask the model for."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        # The token limit can cut an answer short, typically when the model gets
        # stuck repeating itself inside a text value. The fields before that
        # point are still good, so close the object instead of losing them.
        payload = _repair_truncated_json(raw_output)
        if payload is None:
            raise VideoInsightError("the fruit quality could not be determined") from exc
        logger.warning("fruit-quality answer was cut off; recovered %d field(s)", len(payload))
    if not isinstance(payload, dict):
        raise VideoInsightError("the fruit quality could not be determined")
    return payload


TRUNCATED_TEXT_MIN_CHARS = 40
SENTENCE_MARKS = ".؟!۔"


def _repair_truncated_json(raw_output: str) -> dict[str, object] | None:
    """Close a JSON object that ends mid-way and return what it held, if anything."""

    start = raw_output.find("{")
    if start < 0:
        return None
    body = raw_output[start:]
    closers: list[str] = []
    in_string = False
    escaped = False
    # Position of, and open brackets at, the last comma outside a string: the
    # text before it is a sequence of complete members.
    last_member_end: tuple[int, list[str]] | None = None
    string_start = 0
    for index, char in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            string_start = index
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]":
            if not closers or closers.pop() != char:
                return None
            if not closers:
                return None  # complete object: the caller already failed to parse it
        elif char == ",":
            last_member_end = (index, list(closers))

    attempts: list[str] = []
    # A long open string is a sentence that ran away and is worth keeping; a short
    # one is half a keyword ("mo" for "mold") and would only mislead.
    if in_string and len(body) - string_start >= TRUNCATED_TEXT_MIN_CHARS:
        text = body[string_start + 1 :].rstrip("\\")
        sentence_end = max(text.rfind(mark) for mark in SENTENCE_MARKS)
        text = text[: sentence_end + 1] if sentence_end >= 20 else text.rstrip() + "…"
        attempts.append(body[: string_start + 1] + text + '"' + "".join(reversed(closers)))
    if last_member_end is not None:
        index, open_closers = last_member_end
        attempts.append(body[:index] + "".join(reversed(open_closers)))
    for attempt in attempts:
        try:
            payload = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload:
            return payload
    return None


def collapse_repetition(text: str, *, max_phrase_words: int = 12) -> str:
    """Cut a text where it starts repeating the same phrase over and over.

    A small model that loses the thread repeats one phrase until the token limit.
    Everything from the second occurrence on is noise; the sentence before it is
    kept whole when there is one.
    """

    words = text.split()
    for start in range(len(words)):
        for size in range(1, max_phrase_words + 1):
            phrase = words[start : start + size]
            if len(phrase) < size or start + 3 * size > len(words):
                break
            if (
                words[start + size : start + 2 * size] == phrase
                and words[start + 2 * size : start + 3 * size] == phrase
            ):
                kept = " ".join(words[: start + size])
                sentence_end = max(kept.rfind(mark) for mark in SENTENCE_MARKS)
                if sentence_end >= 20:
                    return kept[: sentence_end + 1].strip()
                return kept.rstrip("،,;؛ ") + "…"
    return text


def _parse_fruit_quality(raw_output: str) -> dict[str, object]:
    """Validate the model response before exposing it through the API."""

    try:
        return _fruit_quality_core(_fruit_json_object(raw_output))
    except VideoInsightError:
        # The generated text never leaves the service, but operators need it to
        # see why an answer was rejected.
        logger.warning("rejected fruit-quality answer: %s", raw_output[:600])
        raise


def _fruit_quality_core(payload: dict[str, object]) -> dict[str, object]:
    """Strict contract for the core fields: any violation is an error."""

    has_fruit = payload.get("has_fruit")
    label = payload.get("label")
    summary = payload.get("summary_fa")
    distribution = payload.get("distribution")
    if has_fruit is False:
        # "No fruit here" is a valid answer, not an error: the model has nothing
        # to score, so it usually sends nulls or zeros for the remaining fields.
        return _no_fruit_result(payload)
    if not isinstance(has_fruit, bool) or label not in FRUIT_QUALITY_LABELS:
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
    normalized_distribution = _balanced_distribution(normalized_distribution)
    count = payload.get("fruit_count_estimate")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise VideoInsightError("the fruit quality could not be determined")
    if not has_fruit:
        label = "نامشخص"
        count = None

    core: dict[str, object] = {
        "has_fruit": has_fruit,
        "label": label,
        "freshness_score": freshness_score,
        "distribution": normalized_distribution,
        "fruit_count_estimate": count,
        "confidence": confidence,
    }
    # The numbers are the verdict; the sentence only illustrates them. When the
    # model's sentence is missing or unusable the verdict is put into words.
    core["summary_fa"] = (_summary_text(summary) if isinstance(summary, str) else "") or fruit_verdict_fa(core)
    return core


SUMMARY_MAX_CHARS = 500
# First-person chatter ("I can describe this image in Persian ...") says nothing
# about the fruit. No description of produce needs a first-person verb.
_META_SENTENCE = re.compile(r"توانم|خواهم|زبان فارسی|هوش مصنوعی")
_SENTENCE_SPLIT = re.compile(r"(?<=[.؟!۔])\s+")


def _summary_text(summary: str) -> str:
    """Readable text: no echoed prefix, chatter or repetition loop, bounded length."""

    text = collapse_repetition(clean_model_text(summary))
    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    text = " ".join(part.strip() for part in sentences if not _META_SENTENCE.search(part))
    if len(text) > SUMMARY_MAX_CHARS:
        text = text[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return text


FRUIT_GRADE_LABELS_FA = {"A": "درجه یک", "B": "درجه دو", "C": "درجه سه", "D": "نامرغوب"}
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fruit_verdict_fa(core: dict[str, object]) -> str:
    """The verdict in words, composed from the validated numbers, never from model text."""

    if core.get("has_fruit") is not True:
        return NO_FRUIT_SUMMARY_FA
    score = core["freshness_score"]
    grade_label = FRUIT_GRADE_LABELS_FA.get(fruit_grade(score, True) or "", "")
    distribution = core["distribution"]
    assert isinstance(distribution, dict)
    text = (
        f"کیفیت ظاهری «{core['label']}» ارزیابی شد: امتیاز تازگی {score} از 100"
        f"{f' ({grade_label})' if grade_label else ''}. "
        f"حدود {distribution['fresh']}٪ میوه‌ها تازه، {distribution['middle']}٪ متوسط"
        f" و {distribution['rotten']}٪ فاسد به نظر می‌رسند."
    )
    count = core.get("fruit_count_estimate")
    if isinstance(count, int) and count > 0:
        text += f" تعداد تقریبی میوه‌های قابل مشاهده: {count}."
    return text.translate(_PERSIAN_DIGITS)


NO_FRUIT_SUMMARY_FA = "در تصویر میوه‌ای به‌وضوح دیده نمی‌شود."


def _no_fruit_result(payload: dict[str, object]) -> dict[str, object]:
    confidence = payload.get("confidence")
    summary = payload.get("summary_fa")
    return {
        "has_fruit": False,
        "label": "نامشخص",
        "freshness_score": 0,
        "distribution": {"fresh": 0, "middle": 0, "rotten": 0},
        "fruit_count_estimate": None,
        "confidence": (
            round(confidence)
            if isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 100
            else 0
        ),
        "summary_fa": (
            (_summary_text(summary) if isinstance(summary, str) and summary.strip() else "")
            or NO_FRUIT_SUMMARY_FA
        ),
    }


def _balanced_distribution(values: dict[str, int]) -> dict[str, int]:
    """Make the three shares total exactly 100.

    A small model often misses the total by a few points (for example 60/30/15).
    Within ten points the shares are rescaled with largest-remainder rounding;
    anything further off is not a distribution and is rejected.
    """

    total = sum(values.values())
    if total == 100:
        return values
    if not 90 <= total <= 110:
        raise VideoInsightError("the fruit quality percentages must total 100")
    scaled = {key: value * 100 / total for key, value in values.items()}
    floored = {key: int(share) for key, share in scaled.items()}
    remainder = 100 - sum(floored.values())
    for key in sorted(scaled, key=lambda item: scaled[item] - floored[item], reverse=True)[:remainder]:
        floored[key] += 1
    return floored


def fruit_grade(freshness_score: object, has_fruit: object = True) -> str | None:
    """Commercial grade derived from the score; the model's own grade is never used."""

    if has_fruit is not True:
        return None
    if isinstance(freshness_score, bool) or not isinstance(freshness_score, (int, float)):
        return None
    if freshness_score >= 85:
        return "A"
    if freshness_score >= 70:
        return "B"
    if freshness_score >= 50:
        return "C"
    return "D"


def _optional_percentage(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    rounded = round(value)
    return rounded if 0 <= rounded <= 100 else None


# Small models sometimes echo a field description in front of their answer,
# e.g. "توضیح کوتاه فارسی: ...". Such a prefix carries no information.
_ECHOED_PREFIX = re.compile(
    r"^\s*(?:توضیح|توصیه|یادداشت|خلاصه)(?:\s+(?:خیلی|نگهداری))?\s+کوتاه(?:\s+فارسی)?\s*[:：\-–]\s*"
)


def clean_model_text(value: str) -> str:
    """Strip an echoed instruction prefix and surrounding whitespace."""

    return _ECHOED_PREFIX.sub("", value).strip()


def _optional_text(value: object, limit: int = 500) -> str | None:
    if not isinstance(value, str) or not clean_model_text(value):
        return None
    return collapse_repetition(clean_model_text(value))[:limit]


def empty_fruit_details() -> dict[str, object]:
    """Detail fields of a summary answer: present, but empty."""

    return {
        "fruit_types": [],
        "defects": [],
        "shelf_life_days_estimate": None,
        "recommendation_fa": None,
        "model_recommendation_fa": None,
        "storage_advice_fa": None,
    }


def _fruit_name(value: object) -> str | None:
    """A fruit name is a short noun phrase; a whole sentence in its place is dropped."""

    name = _optional_text(value, 100)
    if name is None or len(name) > 30 or len(name.split()) > 3:
        return None
    if any(mark in name for mark in SENTENCE_MARKS + "،:"):
        return None
    return name


# What a seller should do follows from the grade. It is never taken from the
# model, whose advice can contradict its own score ("sell it" for rotten fruit).
FRUIT_ACTIONS_FA = {
    "A": "محموله برای عرضه عادی مناسب است؛ شرایط نگهداری فعلی حفظ شود.",
    "B": "این محموله زودتر از بارهای تازه‌تر عرضه شود و میوه‌های لک‌دار جدا شوند.",
    "C": "میوه‌های آسیب‌دیده جدا شوند و باقی بار همان روز و با تخفیف عرضه شود.",
    "D": "این محموله از عرضه جمع‌آوری شود؛ میوه‌های فاسد معدوم و باقی‌مانده پیش از فروش سورت شود.",
}


def fruit_action_fa(core: dict[str, object]) -> str | None:
    grade = fruit_grade(core.get("freshness_score"), core.get("has_fruit"))
    return FRUIT_ACTIONS_FA.get(grade) if grade else None


def _parse_fruit_details(payload: dict[str, object]) -> dict[str, object]:
    """Lenient contract for the optional fields: bad entries are dropped, never raised."""

    details = empty_fruit_details()
    fruit_types: list[dict[str, object]] = []
    raw_types = payload.get("fruit_types")
    for item in raw_types if isinstance(raw_types, list) else ():
        if not isinstance(item, dict):
            continue
        name = _fruit_name(item.get("name_fa"))
        share = _optional_percentage(item.get("share_percent"))
        if name is None or share is None:
            continue
        existing = next((entry for entry in fruit_types if entry["name_fa"] == name), None)
        if existing is not None:
            existing["share_percent"] = min(100, int(existing["share_percent"]) + share)
        else:
            fruit_types.append({"name_fa": name, "share_percent": share})
    details["fruit_types"] = fruit_types[:MAX_FRUIT_TYPES]

    defects: list[dict[str, object]] = []
    raw_defects = payload.get("defects")
    for item in raw_defects if isinstance(raw_defects, list) else ():
        if not isinstance(item, dict):
            continue
        raw_type = item.get("type")
        raw_severity = item.get("severity")
        affected = _optional_percentage(item.get("affected_percent"))
        if not isinstance(raw_type, str) or not raw_type.strip() or affected is None:
            continue
        if not isinstance(raw_severity, str) or raw_severity.strip().lower() not in FRUIT_DEFECT_SEVERITIES:
            continue
        defect_type = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        if defect_type not in FRUIT_DEFECT_LABELS_FA:
            # A real but unlisted defect is still worth reporting under "other".
            defect_type = "other"
        defects.append({
            "type": defect_type,
            "label_fa": FRUIT_DEFECT_LABELS_FA[defect_type],
            "severity": raw_severity.strip().lower(),
            "affected_percent": affected,
            "note_fa": _optional_text(item.get("note_fa"), 300) or "",
        })
    details["defects"] = defects[:MAX_FRUIT_DEFECTS]

    shelf_life = payload.get("shelf_life_days_estimate")
    if (
        isinstance(shelf_life, (int, float))
        and not isinstance(shelf_life, bool)
        and shelf_life == shelf_life
        and 0 <= shelf_life <= 365
    ):
        details["shelf_life_days_estimate"] = round(shelf_life)
    details["model_recommendation_fa"] = _optional_text(payload.get("recommendation_fa"))
    details["storage_advice_fa"] = _optional_text(payload.get("storage_advice_fa"))
    return details


def _parse_fruit_frame(raw_output: str) -> dict[str, object]:
    """Validate one single-image answer; a violation fails only that frame."""

    payload = _fruit_json_object(raw_output)
    has_fruit = payload.get("has_fruit")
    label = payload.get("label")
    score = _optional_percentage(payload.get("freshness_score"))
    if (
        not isinstance(has_fruit, bool)
        or not isinstance(label, str)
        or label not in FRUIT_QUALITY_LABELS
        or score is None
    ):
        raise VideoInsightError("the fruit quality of this frame could not be determined")
    return {
        "has_fruit": has_fruit,
        "freshness_score": score,
        "label": label if has_fruit else "نامشخص",
        "note_fa": _optional_text(payload.get("note_fa"), 300) or "",
    }


def frame_statistics(frames: Sequence[dict[str, object]]) -> dict[str, object]:
    """Score spread over the frames that were analysed successfully.

    ``analyzed`` counts frames with a per-frame answer; the score figures use the
    analysed frames that show fruit (population standard deviation).
    """

    analyzed = [
        item for item in frames
        if item.get("error") is None and isinstance(item.get("has_fruit"), bool)
    ]
    scores = [
        float(item["freshness_score"])  # type: ignore[arg-type]
        for item in analyzed
        if item.get("has_fruit") and item.get("freshness_score") is not None
    ]
    if not scores:
        return {
            "analyzed": len(analyzed), "score_min": None, "score_max": None,
            "score_mean": None, "score_stddev": None,
        }
    return {
        "analyzed": len(analyzed),
        "score_min": int(min(scores)),
        "score_max": int(max(scores)),
        "score_mean": round(statistics.fmean(scores), 1),
        "score_stddev": round(statistics.pstdev(scores), 1),
    }


def encode_thumbnail(
    frame: np.ndarray,
    *,
    max_width: int = THUMBNAIL_MAX_WIDTH,
    quality: int = THUMBNAIL_JPEG_QUALITY,
) -> str | None:
    """Encode an RGB frame as a small JPEG data URL; ``None`` if it cannot be encoded."""

    try:
        image = np.ascontiguousarray(frame)
        height, width = image.shape[:2]
        if width > max_width:
            resized_height = max(1, round(height * max_width / width))
            image = cv2.resize(image, (max_width, resized_height), interpolation=cv2.INTER_AREA)
        if image.ndim == 3 and image.shape[2] == 3:
            # Sampled frames are RGB for Qwen; OpenCV encoders expect BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    except (cv2.error, AttributeError, IndexError, TypeError, ValueError):
        return None
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass(frozen=True, slots=True, eq=False)
class SampledFrame:
    """One RGB frame together with its position in the analysed window."""

    image: np.ndarray
    timestamp_seconds: float | None


def extract_image_frame(image_path: str | Path) -> np.ndarray:
    """Decode one still image as RGB for Qwen."""

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise VideoInsightError("could not decode image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_image_frame(image_path: str | Path) -> list[SampledFrame]:
    """A still image is a one-frame window starting at zero seconds."""

    return [SampledFrame(extract_image_frame(image_path), 0.0)]


def extract_video_frames(
    video_path: str | Path,
    *,
    num_frames: int = 8,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> list[np.ndarray]:
    """Return evenly spaced RGB frames from the requested video-time window."""

    return [
        item.image
        for item in sample_video_frames(
            video_path,
            num_frames=num_frames,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
        )
    ]


def sample_video_frames(
    video_path: str | Path,
    *,
    num_frames: int = 8,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> list[SampledFrame]:
    """Evenly spaced RGB frames with their video time (``None`` without a frame rate)."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoInsightError("video does not contain any decodable frames")
        first_index = 0
        last_index = total_frames - 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if window_start_seconds is not None or window_end_seconds is not None:
            if not fps > 0:
                raise VideoInsightError("video frame rate is unavailable")
            first_index = max(0, int((window_start_seconds or 0.0) * fps))
            requested_end = window_end_seconds if window_end_seconds is not None else total_frames / fps
            last_index = min(total_frames - 1, max(first_index, int(requested_end * fps) - 1))
            if first_index >= total_frames:
                raise VideoInsightError("video time window is outside the recording")
        count = min(num_frames, last_index - first_index + 1)
        indexes = np.linspace(first_index, last_index, num=count, dtype=int)
        frames: list[SampledFrame] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                frames.append(SampledFrame(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    round(int(index) / fps, 3) if fps > 0 else None,
                ))
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

    return [
        item.image
        for item in sample_stream_frames(
            stream_url,
            num_frames=num_frames,
            sample_interval_seconds=sample_interval_seconds,
        )
    ]


def sample_stream_frames(
    stream_url: str,
    *,
    num_frames: int = 8,
    sample_interval_seconds: float = 0.75,
) -> list[SampledFrame]:
    """Live RGB frames with the seconds elapsed since the first sampled frame."""

    capture = cv2.VideoCapture(stream_url)
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open camera stream")
        frames: list[SampledFrame] = []
        first_sample: float | None = None
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
            first_sample = now if first_sample is None else first_sample
            frames.append(SampledFrame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(now - first_sample, 3)
            ))
            next_sample = now + sample_interval_seconds
    finally:
        capture.release()
    if not frames:
        raise VideoInsightError("could not capture frames from camera stream")
    return frames


def _is_out_of_memory(torch: Any, error: BaseException) -> bool:
    """CUDA OOM, whether raised as the dedicated type or as a plain RuntimeError."""

    if isinstance(error, VideoInsightError):
        return False
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(error, oom_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _release_gpu_memory(torch: Any) -> None:
    """Return cached CUDA blocks so a smaller retry has room to run."""

    gc.collect()
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not mask the original failure
            logger.debug("torch.cuda.empty_cache() failed", exc_info=True)


def _halve_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Evenly keep half of the frames, in order, and never fewer than one."""

    count = max(1, len(frames) // 2)
    indexes = np.linspace(0, len(frames) - 1, num=count, dtype=int)
    return [frames[int(index)] for index in indexes]


PRECISIONS = ("auto", "bf16", "fp16", "8bit", "4bit")
GIB = 1024**3
# Room kept free next to full-precision weights for the vision tower, the KV
# cache of up to 16 frames and the CUDA context.
FULL_PRECISION_HEADROOM_BYTES = 4 * GIB
# From this much VRAM on, frames keep (almost) their full resolution.
LARGE_GPU_BYTES = 12 * GIB
# Pixel budget per frame. Qwen2.5-VL sees 28x28-pixel patches; its model card
# recommends 256-1280 patches. 1280 patches hold a 1280x720 camera frame
# without downscaling, which is what makes small bruises and mould visible.
LARGE_GPU_PIXELS = (256 * 28 * 28, 1280 * 28 * 28)
SMALL_GPU_PIXELS = (256 * 256, 512 * 512)


def _optional_env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of pixels")
    return value


def resolve_precision(
    requested: str, *, weights_bytes: int, free_bytes: int, bf16_supported: bool
) -> str:
    """Pick the numeric precision the model is loaded with.

    An explicit choice is honoured as it is. ``auto`` never gives quality away
    when the GPU can afford it: the weights stay unquantized whenever they fit
    with headroom, and only a small card falls back to 4-bit.
    """

    full = "bf16" if bf16_supported else "fp16"
    if requested == "bf16" and not bf16_supported:
        return "fp16"
    if requested != "auto":
        return requested
    needed = int(weights_bytes * 1.2) + FULL_PRECISION_HEADROOM_BYTES
    return full if free_bytes >= needed else "4bit"


def resolve_pixel_budget(
    min_pixels: int | None, max_pixels: int | None, *, total_bytes: int
) -> tuple[int, int]:
    """Per-frame pixel budget: explicit settings win, else it follows the GPU size."""

    default_min, default_max = LARGE_GPU_PIXELS if total_bytes >= LARGE_GPU_BYTES else SMALL_GPU_PIXELS
    resolved_max = max_pixels if max_pixels is not None else default_max
    resolved_min = min(min_pixels if min_pixels is not None else default_min, resolved_max)
    return resolved_min, resolved_max


def _weights_bytes(model_path: Path) -> int:
    """Size of the checkpoint on disk, i.e. of the weights at their stored precision."""

    return sum(
        item.stat().st_size
        for pattern in ("*.safetensors", "*.bin")
        for item in model_path.glob(pattern)
    )


class VideoInsightService:
    """Lazily loads one vision-language model and serializes GPU generation calls."""

    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "VIDEO_INSIGHT_MODEL_PATH", "/models/Qwen2.5-VL-3B-Instruct"
        )
        self.requested_precision = os.environ.get("VIDEO_INSIGHT_PRECISION", "").strip().lower() or "auto"
        if self.requested_precision not in PRECISIONS:
            raise ValueError(f"VIDEO_INSIGHT_PRECISION must be one of: {', '.join(PRECISIONS)}")
        # None means "follow the GPU size"; resolved when the model is loaded.
        self.min_pixels = _optional_env_int("VIDEO_INSIGHT_MIN_PIXELS")
        self.max_pixels = _optional_env_int("VIDEO_INSIGHT_MAX_PIXELS")
        self.precision: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def model_name(self) -> str:
        """Public model identifier: the directory name, never the server path."""

        return Path(self.model_path).name or str(self.model_path)

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
            load_options: dict[str, Any] = {"local_files_only": True}
            if torch.cuda.is_available():
                devices = range(torch.cuda.device_count())
                free_bytes = sum(torch.cuda.mem_get_info(index)[0] for index in devices)
                total_bytes = max(torch.cuda.get_device_properties(index).total_memory for index in devices)
                precision = resolve_precision(
                    self.requested_precision,
                    weights_bytes=_weights_bytes(model_path),
                    free_bytes=free_bytes,
                    bf16_supported=torch.cuda.is_bf16_supported(),
                )
                compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                load_options.update(device_map="auto", torch_dtype=compute_dtype)
                if precision == "4bit":
                    load_options["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif precision == "8bit":
                    load_options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                precision, total_bytes = "fp32", 0
                load_options.update(device_map="cpu", torch_dtype=torch.float32)
            self.min_pixels, self.max_pixels = resolve_pixel_budget(
                self.min_pixels, self.max_pixels, total_bytes=total_bytes
            )
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_options
            )
            model.eval()
            self.precision = precision
            logger.info(
                "loaded %s: precision=%s (requested %s), %d-%d pixels per frame",
                self.model_name, precision, self.requested_precision, self.min_pixels, self.max_pixels,
            )
            self._processor = processor
            self._model = model
        return self._model, self._processor, torch

    def describe(self) -> dict[str, object]:
        """What is running, for the health endpoint. Never loads the model."""

        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            "requested_precision": self.requested_precision,
            "precision": self.precision,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

    def interpret(self, frames: list[np.ndarray]) -> dict[str, object]:
        """Return only normalized answers; never expose generated model text."""

        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        raw_output, inference_seconds, used_frames = self._generate_for_frames(
            model,
            processor,
            torch,
            frames,
            (
                "These are consecutive frames from a CCTV video, ordered from oldest to newest. "
                "Answer both questions using only visible evidence. Treat uncertainty as No. "
                "Return exactly one JSON object with no prose or markdown. Each value must be "
                'either "Yes" or "No". Example: '
                '{"fighting":"No","floor_clean":"Yes"}.\n'
                f"1. {FIGHT_QUESTION}\n2. {FLOOR_CLEAN_QUESTION}"
            ),
            max_new_tokens=40,
        )

        return {
            "answers": _parse_yes_no_answers(raw_output),
            "frame_count": len(used_frames),
            "inference_seconds": round(inference_seconds, 3),
        }

    def interpret_video(
        self, frames: list[np.ndarray], *, include_thumbnail: bool = False
    ) -> dict[str, object]:
        """The two fixed answers plus the evidence fields of the HTTP contract."""

        result = self.interpret(frames)
        return {
            **result,
            "thumbnail": encode_thumbnail(frames[len(frames) // 2]) if include_thumbnail else None,
            "model": self.model_name,
        }

    def interpret_fruit_quality(
        self,
        frames: list[np.ndarray],
        *,
        detail_level: str = "summary",
        per_frame: bool = False,
        include_thumbnails: bool = False,
        timestamps: Sequence[float | None] | None = None,
    ) -> dict[str, object]:
        """Score the visible freshness of a fruit image or sampled video."""

        if not frames:
            raise VideoInsightError("at least one frame is required")
        if detail_level not in DETAIL_LEVELS:
            raise ValueError(f"detail_level must be one of: {', '.join(DETAIL_LEVELS)}")
        started = time.perf_counter()
        model, processor, torch = self._load()
        detailed = detail_level == "detailed"
        # A one-word question is far more reliable on a small model than the
        # has_fruit field of the long answer, which it tends to fill with "true"
        # whatever the picture shows. Scenes without produce stop here.
        fruit_visible, presence_seconds = self._fruit_is_visible(model, processor, torch, frames)
        if not fruit_visible:
            return self._fruit_quality_response(
                _no_fruit_result({"confidence": 90}),
                empty_fruit_details(),
                self._frame_entries(
                    model, processor, torch, frames, timestamps,
                    per_frame=False, include_thumbnails=include_thumbnails,
                ),
                detail_level=detail_level,
                frame_count=len(frames),
                inference_seconds=presence_seconds,
                started=started,
            )
        raw_output, inference_seconds, used_frames = self._generate_for_frames(
            model,
            processor,
            torch,
            frames,
            FRUIT_QUALITY_DETAILED_PROMPT if detailed else FRUIT_QUALITY_PROMPT,
            max_new_tokens=700 if detailed else 220,
        )
        inference_seconds += presence_seconds
        details = empty_fruit_details()
        try:
            payload = _fruit_json_object(raw_output)
            core = _fruit_quality_core(payload)
            if detailed:
                details = _parse_fruit_details(payload)
        except VideoInsightError:
            logger.warning("rejected fruit-quality answer: %s", raw_output[:600])
            if not detailed:
                raise
            # The long answer was unusable (truncated or malformed). The core
            # verdict matters most, so ask again with the short, reliable prompt.
            logger.warning("detailed fruit answer was unusable; retrying with the summary prompt")
            raw_output, retry_seconds, used_frames = self._generate_for_frames(
                model, processor, torch, used_frames, FRUIT_QUALITY_PROMPT, max_new_tokens=220
            )
            inference_seconds += retry_seconds
            core = _parse_fruit_quality(raw_output)
        if not core["has_fruit"]:
            details = empty_fruit_details()

        frame_results = self._frame_entries(
            model, processor, torch, frames, timestamps,
            per_frame=per_frame, include_thumbnails=include_thumbnails,
        )
        return self._fruit_quality_response(
            core,
            details,
            frame_results,
            detail_level=detail_level,
            frame_count=len(used_frames),
            inference_seconds=inference_seconds,
            started=started,
        )

    def _fruit_quality_response(
        self,
        core: dict[str, object],
        details: dict[str, object],
        frame_results: list[dict[str, object]],
        *,
        detail_level: str,
        frame_count: int,
        inference_seconds: float,
        started: float,
    ) -> dict[str, object]:
        return {
            **core,
            "verdict_fa": fruit_verdict_fa(core),
            "frame_count": frame_count,
            "inference_seconds": round(inference_seconds, 3),
            "analysis_version": ANALYSIS_VERSION,
            "detail_level": detail_level,
            "model": self.model_name,
            "grade": fruit_grade(core["freshness_score"], core["has_fruit"]),
            **details,
            "recommendation_fa": fruit_action_fa(core),
            "frames": frame_results,
            "frame_statistics": frame_statistics(frame_results),
            "total_seconds": round(time.perf_counter() - started, 3),
        }

    def _frame_entries(
        self,
        model: Any,
        processor: Any,
        torch: Any,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[float | None] | None,
        *,
        per_frame: bool,
        include_thumbnails: bool,
    ) -> list[dict[str, object]]:
        if not (per_frame or include_thumbnails):
            return []
        frame_results: list[dict[str, object]] = []
        for index, frame in enumerate(frames):
            timestamp = timestamps[index] if timestamps is not None and index < len(timestamps) else None
            entry: dict[str, object] = {
                "index": index,
                "timestamp_seconds": timestamp,
                "has_fruit": None,
                "freshness_score": None,
                "label": None,
                "note_fa": None,
                "thumbnail": encode_thumbnail(frame) if include_thumbnails else None,
                "error": None,
            }
            if per_frame:
                entry.update(self._assess_frame(model, processor, torch, frame))
            frame_results.append(entry)
        return frame_results

    def _fruit_is_visible(
        self, model: Any, processor: Any, torch: Any, frames: Sequence[np.ndarray]
    ) -> tuple[bool, float]:
        """Ask the one-word presence question on a few evenly spaced frames.

        Only a clear "No" stops the analysis; an unclear answer lets the full
        prompt decide, so the check can never hide fruit that is there.
        """

        step = max(1, -(-len(frames) // FRUIT_PRESENCE_MAX_FRAMES))
        sample = list(frames[::step])[:FRUIT_PRESENCE_MAX_FRAMES]
        raw_output, seconds, _used = self._generate_for_frames(
            model, processor, torch, sample, FRUIT_PRESENCE_PROMPT, max_new_tokens=4
        )
        answer = re.search(r"\b(yes|no)\b", raw_output.lower())
        if answer is None:
            logger.warning("unclear fruit-presence answer: %s", raw_output[:80])
            return True, seconds
        return answer.group(1) == "yes", seconds

    def _assess_frame(
        self, model: Any, processor: Any, torch: Any, frame: np.ndarray
    ) -> dict[str, object]:
        """Single-image verdict; a failure is reported on the frame, never raised."""

        try:
            raw_output, _seconds, _used = self._generate_for_frames(
                model, processor, torch, [frame], FRUIT_FRAME_PROMPT, max_new_tokens=96
            )
            return _parse_fruit_frame(raw_output)
        except VideoInsightError as exc:
            return {"error": str(exc)}
        except Exception:  # noqa: BLE001 - one bad frame must not fail the request
            logger.exception("per-frame fruit quality analysis failed")
            return {"error": "frame analysis failed"}

    def _generate_for_frames(
        self,
        model: Any,
        processor: Any,
        torch: Any,
        frames: Sequence[np.ndarray],
        prompt_text: str,
        *,
        max_new_tokens: int,
    ) -> tuple[str, float, list[np.ndarray]]:
        """Run one prompt over the frames; on GPU OOM retry once with half of them.

        Returns the raw text, the generation time and the frames actually used.
        """

        attempt_frames = list(frames)
        for attempt in range(2):
            try:
                with self._inference_lock, torch.inference_mode():
                    content: list[dict[str, Any]] = [
                        {"type": "image", "image": frame} for frame in attempt_frames
                    ]
                    content.append({"type": "text", "text": prompt_text})
                    raw_output, inference_seconds = self._generate(
                        model,
                        processor,
                        torch,
                        [{"role": "user", "content": content}],
                        images=attempt_frames,
                        max_new_tokens=max_new_tokens,
                    )
                return raw_output, inference_seconds, attempt_frames
            except Exception as exc:  # noqa: BLE001 - only OOM is handled, the rest re-raised
                if not _is_out_of_memory(torch, exc):
                    raise
                logger.warning(
                    "GPU out of memory while generating with %d frame(s) (attempt %d)",
                    len(attempt_frames), attempt + 1,
                )
            # Outside the except block the traceback no longer pins the failed
            # tensors, so the cache can actually be returned to the driver.
            _release_gpu_memory(torch)
            attempt_frames = _halve_frames(attempt_frames)
        raise VideoInsightError(OUT_OF_MEMORY_MESSAGE)

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

"""Grounded selection of cinematic property details.

VGGT decides which photographs belong to the same room and how the story moves
through the property. This module answers a separate question: which actual
object inside that room deserves a close-up. It deliberately does not ask a
generic image-text model to guess whether a grid crop "looks detailed".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)

OUTPUT_ASPECT = 16 / 9
MIN_SUBJECT_COVERAGE = 0.12
MIN_CROP_WIDTH_FRACTION = 0.18
# Generated video needs a little surrounding context, but the subject remains
# the composition. Ten percent padding left the fireplace test dominated by a
# blank wall; three percent retains the full object without diluting the shot.
SUBJECT_PADDING = 0.03
EDGE_MARGIN_FRACTION = 0.015
DETAIL_METRICS_KEY = "detail_objects_v1"
DETAIL_ANALYSIS_VERSION = 1


@dataclass(frozen=True)
class DetailSubject:
    """An allowed object class and the phrases Grounding DINO may detect."""

    key: str
    label: str
    prompts: tuple[str, ...]
    priority: float = 1.0


# This is the product's editable detail catalogue. Room type is only scope for
# valid objects; it never makes a photo or crop desirable by itself. Additions
# from editorial review belong here with a real subject label and prompt.
DETAIL_SUBJECTS_BY_ROOM: dict[str, tuple[DetailSubject, ...]] = {
    "entrance": (
        DetailSubject("entry_art", "Wall art", ("wall art",), 0.75),
        DetailSubject("entry_light", "Light fixture", ("light fixture",), 0.80),
    ),
    "kitchen": (
        DetailSubject("cooking", "Stove or oven", ("stove", "oven", "cooktop"), 1.00),
        DetailSubject("faucet", "Kitchen faucet", ("faucet",), 0.85),
        DetailSubject("range_hood", "Range hood", ("range hood",), 0.75),
        DetailSubject("microwave", "Microwave", ("microwave",), 0.60),
    ),
    "living room": (
        DetailSubject("fireplace", "Fireplace", ("fireplace", "mantle"), 1.00),
        DetailSubject("living_art", "Wall art", ("wall art",), 0.70),
        DetailSubject("living_light", "Light fixture", ("light fixture",), 0.80),
    ),
    "dining room": (
        DetailSubject("dining_light", "Light fixture", ("chandelier", "light fixture"), 0.90),
        DetailSubject("dining_art", "Wall art", ("wall art",), 0.70),
    ),
    "patio": (
        DetailSubject("grill", "Barbecue grill", ("barbecue grill",), 1.00),
        DetailSubject("fire_pit", "Fire pit", ("fire pit",), 0.90),
        DetailSubject("pool", "Pool", ("pool",), 0.85),
    ),
    "pool": (
        DetailSubject("pool", "Pool", ("pool",), 1.00),
        DetailSubject("fire_pit", "Fire pit", ("fire pit",), 0.90),
    ),
    "backyard": (
        DetailSubject("fire_pit", "Fire pit", ("fire pit",), 0.90),
        DetailSubject("pool", "Pool", ("pool",), 0.90),
        DetailSubject("grill", "Barbecue grill", ("barbecue grill",), 0.85),
    ),
    "exterior back": (
        DetailSubject("grill", "Barbecue grill", ("barbecue grill",), 0.95),
        DetailSubject("pool", "Pool", ("pool",), 0.90),
        DetailSubject("fire_pit", "Fire pit", ("fire pit",), 0.90),
    ),
    "amenities": (
        DetailSubject("fireplace", "Fireplace", ("fireplace", "mantle"), 1.00),
        DetailSubject("pool", "Pool", ("pool",), 0.90),
        DetailSubject("light", "Light fixture", ("light fixture",), 0.75),
    ),
}


def detail_subjects(room_type: str | None) -> tuple[DetailSubject, ...]:
    """Return the whitelist of subjects that can earn a detail in this room."""
    return DETAIL_SUBJECTS_BY_ROOM.get((room_type or "").strip().lower(), ())


def is_detail_eligible(room_type: str | None) -> bool:
    """A room is eligible only when it has named, worthwhile object classes."""
    return bool(detail_subjects(room_type))


@dataclass(frozen=True)
class DetailCrop:
    """A 16:9 crop around a verified subject, in source-image pixels."""

    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class ResolvedDetail:
    """One complete, detected object ready to insert after its room wide."""

    source_photo_id: int
    subject_key: str
    subject_label: str
    subject_box: tuple[int, int, int, int]
    detector_score: float
    crop: DetailCrop | None
    source_size: tuple[int, int]
    # A dedicated object photograph is already a stronger composition than a
    # synthetic landscape crop. `source_frame` tells the renderer to preserve it.
    framing: str = "object_crop"
    selection_score: float = 0.0

    @property
    def score(self) -> float:
        return self.selection_score or (self.crop.score if self.crop else self.detector_score)


@dataclass(frozen=True)
class _Detection:
    subject: DetailSubject
    score: float
    box: tuple[float, float, float, float]


def _subject_for_label(label: str, subjects: Sequence[DetailSubject]) -> DetailSubject | None:
    normalized = label.strip().lower()
    for subject in subjects:
        for prompt in subject.prompts:
            if prompt in normalized or normalized in prompt:
                return subject
    return None


def _box_is_complete(box: Sequence[float], source_size: tuple[int, int]) -> bool:
    """Reject a detector box that touches an image boundary and is likely cut off."""
    left, top, right, bottom = box
    width, height = source_size
    margin_x = width * EDGE_MARGIN_FRACTION
    margin_y = height * EDGE_MARGIN_FRACTION
    return (
        left >= margin_x
        and top >= margin_y
        and right <= width - margin_x
        and bottom <= height - margin_y
    )


def crop_for_subject(
    box: Sequence[float],
    source_size: tuple[int, int],
    *,
    score: float,
) -> DetailCrop | None:
    """Build a centered 16:9 crop that keeps a complete subject meaningfully large."""
    if not _box_is_complete(box, source_size):
        return None

    left, top, right, bottom = box
    source_width, source_height = source_size
    subject_width = right - left
    subject_height = bottom - top
    if subject_width <= 0 or subject_height <= 0:
        return None

    padded_width = subject_width * (1 + 2 * SUBJECT_PADDING)
    padded_height = subject_height * (1 + 2 * SUBJECT_PADDING)
    crop_width = max(padded_width, padded_height * OUTPUT_ASPECT)
    crop_height = crop_width / OUTPUT_ASPECT
    if crop_width > source_width:
        crop_width = float(source_width)
        crop_height = crop_width / OUTPUT_ASPECT
    if crop_height > source_height:
        crop_height = float(source_height)
        crop_width = crop_height * OUTPUT_ASPECT
    if crop_width < subject_width or crop_height < subject_height:
        return None
    if crop_width < source_width * MIN_CROP_WIDTH_FRACTION:
        # A tiny detected box can be technically complete yet still produce a
        # postage-stamp crop of a false positive. Premium detail shots need
        # enough visual context to be a usable generated-video keyframe.
        return None

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = max(0.0, min(source_width - crop_width, center_x - crop_width / 2))
    crop_top = max(0.0, min(source_height - crop_height, center_y - crop_height / 2))
    crop_area = crop_width * crop_height
    coverage = (subject_width * subject_height) / crop_area if crop_area else 0.0
    if coverage < MIN_SUBJECT_COVERAGE:
        return None

    return DetailCrop(
        x=round(crop_left),
        y=round(crop_top),
        width=round(crop_width),
        height=round(crop_height),
        score=float(score),
    )


def _uses_source_frame(box: Sequence[float], source_size: tuple[int, int]) -> bool:
    """Recognise a photo deliberately composed as a vertical object detail.

    This is intentionally strict: a broad kitchen photo containing a stove still
    earns a crop, while a portrait image dominated by a fixture or artwork keeps
    its authored composition intact.
    """
    left, top, right, bottom = box
    width, height = source_size
    if height <= width * 1.12:
        return False
    subject_width_fraction = (right - left) / width
    subject_height_fraction = (bottom - top) / height
    subject_center_x = ((left + right) / 2) / width
    return (
        subject_width_fraction >= 0.12
        and subject_height_fraction >= 0.35
        and 0.18 <= subject_center_x <= 0.82
    )


def _resolved_detection(
    photo_id: int,
    source_size: tuple[int, int],
    detection: _Detection,
) -> ResolvedDetail | None:
    ranking_score = detection.score * detection.subject.priority
    framing = "source_frame" if _uses_source_frame(detection.box, source_size) else "object_crop"
    if framing == "source_frame":
        # A tall, intentionally composed detail cannot fit a 16:9 crop without
        # cutting off its subject. Validate the same complete detector box, then
        # preserve the original frame instead of rejecting a good photograph.
        if not _box_is_complete(detection.box, source_size):
            return None
        return ResolvedDetail(
            source_photo_id=photo_id,
            subject_key=detection.subject.key,
            subject_label=detection.subject.label,
            subject_box=tuple(round(value) for value in detection.box),
            detector_score=detection.score,
            crop=None,
            source_size=source_size,
            framing=framing,
            selection_score=ranking_score,
        )
    crop = crop_for_subject(detection.box, source_size, score=ranking_score)
    if crop is None:
        return None
    return ResolvedDetail(
        source_photo_id=photo_id,
        subject_key=detection.subject.key,
        subject_label=detection.subject.label,
        subject_box=tuple(round(value) for value in detection.box),
        detector_score=detection.score,
        crop=crop,
        source_size=source_size,
        framing=framing,
        selection_score=ranking_score,
    )


class _GroundingDinoDetector:
    """Lazy, one-copy Grounding DINO runtime for object-box detection."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.utils import logging as transformers_logging

        try:
            snapshot = snapshot_download(
                settings.DETAIL_DETECTOR_MODEL,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO is not cached. Run `hf download "
                f"{settings.DETAIL_DETECTOR_MODEL}` while preparing this worker."
            ) from exc

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        transformers_logging.disable_progress_bar()
        self._processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            snapshot, local_files_only=True,
        ).to(self._device).eval()
        logger.info("Loaded %s detail detector on %s", settings.DETAIL_DETECTOR_MODEL, self._device)

    def detect(
        self,
        images,
        subjects: Sequence[DetailSubject],
    ) -> list[list[_Detection]]:
        """Return prompt-matched detections for an image batch."""
        import torch

        self._ensure_loaded()
        prompts = tuple(dict.fromkeys(
            prompt for subject in subjects for prompt in subject.prompts
        ))
        text = ". ".join(prompts) + "."
        inputs = self._processor(
            images=images,
            text=[text] * len(images),
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        raw_results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=settings.DETAIL_DETECTOR_MIN_SCORE,
            text_threshold=settings.DETAIL_DETECTOR_MIN_SCORE,
            target_sizes=[(image.height, image.width) for image in images],
        )

        detections: list[list[_Detection]] = []
        for result in raw_results:
            image_detections = []
            for score, label, box in zip(
                result["scores"], result["text_labels"], result["boxes"],
            ):
                subject = _subject_for_label(str(label), subjects)
                if subject is not None:
                    image_detections.append(_Detection(
                        subject=subject,
                        score=float(score),
                        box=tuple(float(value) for value in box.tolist()),
                    ))
            detections.append(image_detections)
        return detections

    def release(self) -> None:
        """Drop optional detector weights after a job; VGGT stays the warm model."""
        if self._model is None:
            return
        import gc
        import torch

        self._model = None
        self._processor = None
        device = self._device
        self._device = None
        gc.collect()
        if device is not None and getattr(device, "type", "") == "cuda":
            torch.cuda.empty_cache()
        elif device is not None and getattr(device, "type", "") == "mps":
            torch.mps.empty_cache()


detail_detector = _GroundingDinoDetector()


def _stored_detail(detail: ResolvedDetail) -> dict[str, Any]:
    return {
        "subject_key": detail.subject_key,
        "subject_label": detail.subject_label,
        "subject_box": list(detail.subject_box),
        "detector_score": detail.detector_score,
        "crop_box": list(detail.crop.box) if detail.crop else None,
        "selection_score": detail.score,
        "source_size": list(detail.source_size),
        "framing": detail.framing,
    }


def _detail_from_stored(photo_id: int, payload: Mapping[str, Any]) -> ResolvedDetail | None:
    try:
        source_size = tuple(int(value) for value in payload["source_size"])
        subject_box = tuple(int(value) for value in payload["subject_box"])
        crop_box = payload.get("crop_box")
        crop = None
        if crop_box is not None:
            left, top, right, bottom = (int(value) for value in crop_box)
            crop = DetailCrop(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                score=float(payload.get("selection_score") or 0.0),
            )
        return ResolvedDetail(
            source_photo_id=photo_id,
            subject_key=str(payload["subject_key"]),
            subject_label=str(payload["subject_label"]),
            subject_box=subject_box,
            detector_score=float(payload["detector_score"]),
            crop=crop,
            source_size=source_size,
            framing=str(payload.get("framing") or "object_crop"),
            selection_score=float(payload.get("selection_score") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("Ignoring malformed persisted detail for photo %s", photo_id)
        return None


def _resolve_candidates(candidates: Mapping[str, Sequence[ResolvedDetail]]) -> dict[str, list[ResolvedDetail]]:
    """Keep one strongest complete source per room/object combination."""
    resolved: dict[str, list[ResolvedDetail]] = {}
    for room_id, room_candidates in candidates.items():
        best_by_subject: dict[str, ResolvedDetail] = {}
        for candidate in room_candidates:
            current = best_by_subject.get(candidate.subject_key)
            if current is None or (
                candidate.score,
                candidate.detector_score,
                -candidate.source_photo_id,
            ) > (
                current.score,
                current.detector_score,
                -current.source_photo_id,
            ):
                best_by_subject[candidate.subject_key] = candidate
        resolved[room_id] = sorted(
            best_by_subject.values(),
            key=lambda candidate: (-candidate.score, -candidate.detector_score, candidate.source_photo_id),
        )
    return resolved


def resolve_persisted_detail_candidates(
    shots,
    room_photo_ids: Mapping[str, Sequence[int]],
    metric_by_photo: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[ResolvedDetail]]:
    """Select saved Phase-1 detections for the rooms that survived editing.

    This does no network I/O and does not load Grounding DINO. A page refresh is
    therefore a cheap review operation, not hidden inference.
    """
    selected_rooms = {
        shot.room_id: shot.room_type
        for shot in shots
        if not getattr(shot, "is_close_up", False)
    }
    candidates: dict[str, list[ResolvedDetail]] = defaultdict(list)
    for room_id, room_type in selected_rooms.items():
        if not is_detail_eligible(room_type):
            continue
        for photo_id in dict.fromkeys(room_photo_ids.get(room_id, ())):
            stored = (metric_by_photo.get(photo_id, {}).get(DETAIL_METRICS_KEY) or {})
            if stored.get("version") != DETAIL_ANALYSIS_VERSION:
                continue
            for payload in stored.get("objects") or ():
                detail = _detail_from_stored(int(photo_id), payload)
                if detail is not None:
                    candidates[room_id].append(detail)
    return _resolve_candidates(candidates)


def detect_and_persist_photo_details(db, job_id: int, photos, s3_client) -> int:
    """Run optional object detection once during Phase 1 and persist its evidence.

    Details are editorial metadata only: failures must not fail VGGT analysis or
    turn the API endpoint into a second inference worker. The detector is released
    once saved so repeated listings do not retain a second large model in memory.
    """
    from app.db.models import PhotoSceneGeometry

    geometry_rows = {
        int(row.photo_id): row
        for row in db.query(PhotoSceneGeometry).filter(PhotoSceneGeometry.job_id == job_id).all()
    }
    eligible_by_type: dict[str, list[Any]] = defaultdict(list)
    for photo in photos:
        if photo.id not in geometry_rows or photo.exclude or photo.is_duplicate:
            continue
        room_type = (photo.room_override or photo.room_label or "").strip().lower()
        if detail_subjects(room_type):
            eligible_by_type[room_type].append(photo)

    by_photo: dict[int, list[ResolvedDetail]] = defaultdict(list)
    failures = False
    batch_size = max(1, settings.DETAIL_DETECTOR_BATCH_SIZE)
    try:
        for room_type, candidates in eligible_by_type.items():
            subjects = detail_subjects(room_type)
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                decoded: list[tuple[Any, Any]] = []
                try:
                    for photo in batch:
                        try:
                            decoded.append((photo, s3_client.download_image(photo.s3_uri)))
                        except Exception:
                            logger.warning("Could not download detail candidate %s", photo.id, exc_info=True)
                            failures = True
                    decoded = [(photo, image) for photo, image in decoded if image is not None]
                    if not decoded:
                        continue
                    detections = detail_detector.detect([image for _, image in decoded], subjects)
                    for (photo, image), photo_detections in zip(decoded, detections):
                        for detection in photo_detections:
                            detail = _resolved_detection(int(photo.id), image.size, detection)
                            if detail is not None:
                                by_photo[int(photo.id)].append(detail)
                finally:
                    for _, image in decoded:
                        image.close()
    except Exception:
        failures = True
        logger.exception("Detail detection unavailable for Phase-1 job %s", job_id)
    finally:
        detail_detector.release()

    for photo_id, row in geometry_rows.items():
        metrics = dict(row.local_metrics or {})
        details = _resolve_candidates({"photo": by_photo.get(photo_id, [])}).get("photo", [])
        metrics[DETAIL_METRICS_KEY] = {
            "version": DETAIL_ANALYSIS_VERSION,
            "status": "partial" if failures else "complete",
            "objects": [_stored_detail(detail) for detail in details],
        }
        row.local_metrics = metrics
    return sum(len(details) for details in by_photo.values())

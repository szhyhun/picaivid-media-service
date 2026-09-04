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
from typing import Mapping, Sequence

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
    crop: DetailCrop
    source_size: tuple[int, int]


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


detail_detector = _GroundingDinoDetector()


def resolve_detail_candidates(
    shots,
    room_photo_ids: Mapping[str, Sequence[int]],
    photo_uri: Mapping[int, str],
) -> dict[str, list[ResolvedDetail]]:
    """Detect the best allowed subjects across every photo in each selected room.

    A detail's source does not have to be one of the story shot's keyframes.
    That separation prevents a wide with bar stools from displacing another
    kitchen image that contains the actual stove.
    """
    from app.services.storage.s3_client import s3_client

    selected_rooms: dict[str, str] = {}
    for shot in shots:
        if not getattr(shot, "is_close_up", False):
            selected_rooms.setdefault(shot.room_id, shot.room_type)

    candidates: dict[str, list[ResolvedDetail]] = defaultdict(list)
    batch_size = max(1, settings.DETAIL_DETECTOR_BATCH_SIZE)
    for room_id, room_type in selected_rooms.items():
        subjects = detail_subjects(room_type)
        if not subjects:
            continue
        photo_ids = list(dict.fromkeys(room_photo_ids.get(room_id, ())))
        for start in range(0, len(photo_ids), batch_size):
            batch_ids = photo_ids[start:start + batch_size]
            decoded: list[tuple[int, object]] = []
            try:
                for photo_id in batch_ids:
                    uri = photo_uri.get(photo_id)
                    if uri is None:
                        continue
                    try:
                        image = s3_client.download_image(uri)
                    except Exception:
                        logger.warning("Could not download detail candidate %s", photo_id, exc_info=True)
                        continue
                    if image is not None:
                        decoded.append((photo_id, image))
                if not decoded:
                    continue
                detected = detail_detector.detect(
                    [image for _, image in decoded], subjects,
                )
                for (photo_id, image), detections in zip(decoded, detected):
                    for detection in detections:
                        ranking_score = detection.score * detection.subject.priority
                        crop = crop_for_subject(
                            detection.box,
                            image.size,
                            score=ranking_score,
                        )
                        if crop is None:
                            continue
                        candidates[room_id].append(ResolvedDetail(
                            source_photo_id=photo_id,
                            subject_key=detection.subject.key,
                            subject_label=detection.subject.label,
                            subject_box=tuple(round(value) for value in detection.box),
                            detector_score=detection.score,
                            crop=crop,
                            source_size=image.size,
                        ))
            finally:
                for _, image in decoded:
                    image.close()

    # One strongest, complete source per object type. "Stove" and "oven" are
    # intentionally one `cooking` subject, so overlapping boxes cannot create
    # duplicate shots of the same appliance.
    resolved: dict[str, list[ResolvedDetail]] = {}
    for room_id, room_candidates in candidates.items():
        best_by_subject: dict[str, ResolvedDetail] = {}
        for candidate in room_candidates:
            current = best_by_subject.get(candidate.subject_key)
            if current is None or (
                candidate.crop.score,
                candidate.detector_score,
                -candidate.source_photo_id,
            ) > (
                current.crop.score,
                current.detector_score,
                -current.source_photo_id,
            ):
                best_by_subject[candidate.subject_key] = candidate
        resolved[room_id] = sorted(
            best_by_subject.values(),
            key=lambda candidate: (
                -candidate.crop.score,
                -candidate.detector_score,
                candidate.source_photo_id,
            ),
        )
    return resolved

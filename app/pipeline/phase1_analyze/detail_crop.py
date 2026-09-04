"""Manufacture detail shots by cropping into room photographs.

A detail must belong to the room it follows. The reference films cut from a wide
living room to a close-up of *that* living room's fireplace -- the close-up is
the same space seen tighter, which is what makes the pair read as one continuous
look rather than two unrelated pictures. So a detail is always sourced from a
photograph of the room whose beat it accompanies, ideally the very frame that
preceded it.

That also bounds the work. Scoring twenty candidate windows per photograph
across a whole listing is ~5,900 CLIP image encodes, nearly all of them spent on
photographs that never reach the film. Only shots already selected into the shot
line, and their room-mates, are worth searching.

The reference films open on a garden detail and punctuate rooms with close-ups
of flowers and fixtures. Real-estate shoots do not contain such photographs:
scored across all six labelled listings, **2 of 295 photos (0.7%)** read as a
detail rather than a room. Detail shots therefore have to be made, not found.

A crop is scored by how much more it looks like an appealing close-up than like
a blank wall or an out-of-focus area, using the same CLIP embedding the pipeline
already computes. The best window in a frame consistently beats that frame's
average, and visual inspection of the winners returns flowering shrubs,
hydrangeas beside a front door, and foliage -- the exact subjects the reference
films use.

Resolution: source photographs are typically 2048x1365 and output is 1080p.
A 35% crop is roughly 716px wide: tight enough to read as a different visual
beat, while retaining meaningful visual context for a later video model. The
renderer must validate that conditioning resolution when it is selected; this
planner does not assume an untested model-specific lower limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# What a good detail shot looks like, and what it must not look like. The
# difference of the two is the score, so a crop has to be positively appealing
# rather than merely not-blank.
DETAIL_PROMPTS = (
    "a beautiful close-up detail of a decorative object",
    "a close-up of flowers or a plant",
    "an appealing close-up of a design feature or texture",
    "a close-up of a fireplace, faucet, tap or light fixture",
)
NON_DETAIL_PROMPTS = (
    "a blank empty wall",
    "an out of focus blurry area",
    "a plain ceiling",
    "a plain floor",
    "an ordinary wide room photograph",
    "an exposed mechanical fixture, pipe or hardware",
    "an exterior awning, roof or canopy mechanism",
    "an unstyled utility area",
)

# Fraction of the source width one crop spans. See the resolution note above.
CROP_WIDTH_FRACTION = 0.35
OUTPUT_ASPECT = 16 / 9

# Where to look. A coarse grid is enough -- detail subjects are large relative to
# the crop, and a finer search mostly returns near-duplicates of the same window.
_X_POSITIONS = (0.12, 0.32, 0.52, 0.72, 0.88)
_Y_POSITIONS = (0.15, 0.35, 0.55, 0.75)

# A frame with nothing worth cropping into should yield no detail at all rather
# than the least-bad blank wall. Calibrated against the six labelled listings:
# exteriors with planting score +0.01..+0.04, plain interiors score negative.
MIN_DETAIL_SCORE = 0.005

# A room type is a conservative eligibility gate, not proof that a close-up is
# worthwhile. The reference films include bathrooms and bedrooms, but this
# product cannot yet identify a genuinely premium bathroom fixture or bedroom
# object reliably. A generic crop there is usually a wall, bedspread, shower
# panel, or towel. Keep them out until a grounded object scorer earns an
# exception. The rate only breaks ties between *validated* candidates.
CLOSE_UP_RATE_BY_ROOM: dict[str, float] = {
    "entrance": 0.20,
    "kitchen": 0.40,
    "living room": 0.25,
    "dining room": 0.20,
    "patio": 0.35,
    "pool": 0.35,
    "backyard": 0.35,
    "exterior back": 0.30,
    "amenities": 0.25,
    # Not eligible without a future object-specific exception.
    "bedroom": 0.0,
    "bathroom": 0.0,
    "hallway": 0.0,
    "shared space": 0.0,
    "office": 0.0,
    "garage": 0.0,
    "storage": 0.0,
    "laundry room": 0.0,
    "aerial view": 0.0,
    "exterior front": 0.0,
    "front yard": 0.0,
    "building exterior": 0.0,
}
DEFAULT_CLOSE_UP_RATE = 0.0


def close_up_rate(room_type: str | None) -> float:
    """How often this kind of room earns a close-up, from the reference films."""
    return CLOSE_UP_RATE_BY_ROOM.get(
        (room_type or "").strip().lower(), DEFAULT_CLOSE_UP_RATE)


@dataclass(frozen=True)
class DetailCrop:
    """A crop box in source-pixel coordinates, with the score that chose it."""

    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(left, upper, right, lower), ready for PIL's crop()."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class ResolvedDetail:
    """A crop that passed visual scoring for one selected wide shot."""

    crop: DetailCrop
    source_size: tuple[int, int]


def candidate_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Grid of candidate crop windows for a frame of this size."""
    crop_w = int(width * CROP_WIDTH_FRACTION)
    crop_h = int(crop_w / OUTPUT_ASPECT)
    if crop_h > height:
        crop_h = height
        crop_w = int(crop_h * OUTPUT_ASPECT)
    boxes = []
    for fy in _Y_POSITIONS:
        for fx in _X_POSITIONS:
            x = max(0, min(width - crop_w, int((width - crop_w) * fx)))
            y = max(0, min(height - crop_h, int((height - crop_h) * fy)))
            boxes.append((x, y, crop_w, crop_h))
    return boxes


def score_crops(
    crop_embeddings: np.ndarray,
    detail_centroid: np.ndarray,
    non_detail_centroid: np.ndarray,
) -> np.ndarray:
    """How much more each crop reads as an appealing detail than as a blank area.

    Embeddings are assumed L2-normalised, as the pipeline stores them.
    """
    return crop_embeddings @ detail_centroid - crop_embeddings @ non_detail_centroid


def best_crop(
    boxes: list[tuple[int, int, int, int]],
    scores: np.ndarray,
    *,
    min_score: float = MIN_DETAIL_SCORE,
) -> DetailCrop | None:
    """Pick the strongest crop, or nothing if the frame has no detail worth using.

    Returning None matters: a film should skip a detail beat rather than cut to
    a close-up of a blank wall.
    """
    if not boxes or scores.size == 0:
        return None
    index = int(np.argmax(scores))
    if float(scores[index]) < min_score:
        return None
    x, y, w, h = boxes[index]
    return DetailCrop(x=x, y=y, width=w, height=h, score=float(scores[index]))


def find_detail(image, model, preprocess, detail_centroid, non_detail_centroid,
                *, min_score: float = MIN_DETAIL_SCORE) -> DetailCrop | None:
    """Locate the best detail crop in a PIL image, if it has one.

    Kept separate from the scoring maths above so the decision logic stays
    testable without a model or an image.
    """
    import torch

    boxes = candidate_boxes(*image.size)
    crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
    try:
        batch = torch.stack([preprocess(c) for c in crops])
    finally:
        for crop in crops:
            crop.close()
    device = getattr(model, "device", None) or next(model.parameters()).device
    with torch.no_grad():
        features = model.encode_image(batch.to(device))
        features = features / features.norm(dim=-1, keepdim=True)
    scores = score_crops(features.cpu().numpy(), detail_centroid, non_detail_centroid)
    return best_crop(boxes, scores, min_score=min_score)


def resolve_detail_candidates(shots, photo_uri, *, min_score: float = MIN_DETAIL_SCORE):
    """Score every eligible wide shot before any close-up is scheduled.

    `shots` must be the selected wide line, not every uploaded photo. The return
    value maps a wide shot index to the validated crop that can follow it. This
    order matters: selecting positions first and testing crops afterwards left
    holes in the edit whenever a chosen source had no real detail.
    """
    import torch

    from app.models.openclip import openclip_model
    from app.services.storage.s3_client import s3_client

    candidates = [s for s in shots
                  if not getattr(s, "is_close_up", False)
                  and close_up_rate(getattr(s, "room_type", "")) > 0]
    if not candidates:
        return {}

    openclip_model._ensure_loaded()
    model = openclip_model._model
    preprocess = openclip_model._preprocess
    tokenizer = openclip_model._tokenizer
    # The singleton keeps the model on MPS where available, so inputs have to be
    # moved to match or torch raises "Placeholder storage has not been allocated".
    device = openclip_model._device

    def centroid(prompts):
        with torch.no_grad():
            e = model.encode_text(tokenizer(list(prompts)).to(device))
            e = e / e.norm(dim=-1, keepdim=True)
            f = e.mean(0, keepdim=True)
            return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    detail_c = centroid(DETAIL_PROMPTS)
    plain_c = centroid(NON_DETAIL_PROMPTS)

    # Retain crop decisions, never decoded full-resolution images. A 50-photo
    # listing can otherwise leave hundreds of MB of PIL buffers live while the
    # review request is running.
    cache: dict[int, ResolvedDetail | None] = {}
    resolved: dict[int, ResolvedDetail] = {}
    for shot in candidates:
        photo_id = shot.photo_ids[0]
        if photo_id not in cache:
            uri = photo_uri.get(photo_id)
            try:
                image = s3_client.download_image(uri) if uri else None
                if image is None:
                    cache[photo_id] = None
                else:
                    try:
                        crop = find_detail(
                            image, model, preprocess, detail_c, plain_c,
                            min_score=min_score,
                        )
                        cache[photo_id] = (
                            ResolvedDetail(crop=crop, source_size=image.size)
                            if crop is not None else None
                        )
                    finally:
                        image.close()
            except Exception:
                cache[photo_id] = None
        detail = cache[photo_id]
        if detail is not None:
            resolved[int(shot.index)] = detail
    return resolved

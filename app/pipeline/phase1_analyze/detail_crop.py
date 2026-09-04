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

Resolution: source photographs are 2048x1365 and output is 1080p, so a native
crop could only be ~94% of the width, which is not a detail. Generative video
models regenerate pixels at their own resolution, so a tighter crop is
acceptable as *conditioning*; 45% of width is about 920px, enough to condition a
generator while reading as a genuine close-up. Tighter than this and the model
has too little to work from, which is where hallucination starts.
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
)

# Fraction of the source width one crop spans. See the resolution note above.
CROP_WIDTH_FRACTION = 0.45
OUTPUT_ASPECT = 16 / 9

# Where to look. A coarse grid is enough -- detail subjects are large relative to
# the crop, and a finer search mostly returns near-duplicates of the same window.
_X_POSITIONS = (0.12, 0.32, 0.52, 0.72, 0.88)
_Y_POSITIONS = (0.15, 0.35, 0.55, 0.75)

# A frame with nothing worth cropping into should yield no detail at all rather
# than the least-bad blank wall. Calibrated against the six labelled listings:
# exteriors with planting score +0.01..+0.04, plain interiors score negative.
MIN_DETAIL_SCORE = 0.005


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
    with torch.no_grad():
        features = model.encode_image(torch.stack([preprocess(c) for c in crops]))
        features = features / features.norm(dim=-1, keepdim=True)
    scores = score_crops(features.cpu().numpy(), detail_centroid, non_detail_centroid)
    return best_crop(boxes, scores, min_score=min_score)


def details_for_rooms(
    room_photos: dict[str, list[int]],
    image_loader,
    model,
    preprocess,
    detail_centroid: np.ndarray,
    non_detail_centroid: np.ndarray,
    *,
    prefer: dict[str, list[int]] | None = None,
    min_score: float = MIN_DETAIL_SCORE,
) -> dict[str, tuple[int, DetailCrop]]:
    """Find one detail per room, sourced from that room's own photographs.

    `room_photos` maps a room id to the photo ids that belong to it, and
    `prefer` optionally names the photos already chosen for that room's wide
    shots -- those are searched first, because a close-up cut from the very frame
    the viewer just saw reads as the same look continuing rather than as a jump
    to somewhere new.

    Returns room id -> (source photo id, crop). Rooms whose photographs contain
    nothing worth cropping into are simply absent, so the planner schedules no
    detail beat for them.
    """
    import torch

    found: dict[str, tuple[int, DetailCrop]] = {}
    for room_id, photo_ids in room_photos.items():
        ordered = list(dict.fromkeys((prefer or {}).get(room_id, []) + list(photo_ids)))
        best: tuple[int, DetailCrop] | None = None
        for photo_id in ordered:
            image = image_loader(photo_id)
            if image is None:
                continue
            boxes = candidate_boxes(*image.size)
            crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
            with torch.no_grad():
                features = model.encode_image(
                    torch.stack([preprocess(c) for c in crops]))
                features = features / features.norm(dim=-1, keepdim=True)
            scores = score_crops(features.cpu().numpy(),
                                 detail_centroid, non_detail_centroid)
            crop = best_crop(boxes, scores, min_score=min_score)
            if crop is not None and (best is None or crop.score > best[1].score):
                best = (photo_id, crop)
            # A preferred frame that already yields a good detail wins outright:
            # continuity with the shot just seen beats a marginally prettier crop
            # from elsewhere in the room.
            if best is not None and photo_id in (prefer or {}).get(room_id, []):
                break
        if best is not None:
            found[room_id] = best
    return found

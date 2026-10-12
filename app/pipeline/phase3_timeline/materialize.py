"""Turn the computed shot line into a stored, editable timeline.

Until now the shot line was recomputed on every request and identified only by
list position, so nothing could be edited: there was nothing to edit. This
module writes it down once, giving every shot a `shot_uid` that survives
reordering, regeneration and the move from storyboard to edit.

Re-materialising never overwrites in place. It creates a new `version`, because
a user who has spent time rearranging a timeline should not lose it to a
background re-analysis.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.db.models import PhotoSceneGeometry
from app.db.models.timeline import (
    SOURCE_SECONDS,
    Timeline,
    TimelineClip,
    source_seconds_for,
)
from app.pipeline.phase1_analyze.detail_crop import (
    is_detail_eligible,
    resolve_persisted_detail_candidates,
)
from app.pipeline.phase1_analyze.shot_line import (
    build_shot_line,
    schedule_resolved_close_ups,
)


def build_line(db: Session, rooms, *, target_seconds: float, bpm: float):
    """Compute the shot line including close-ups resolved from stored evidence.

    Close-ups are found during Phase 1 and saved; this only *selects* from that
    evidence. It never loads a detector or downloads a source image, which is
    what keeps the interactive endpoints fast.
    """
    line = build_shot_line(
        rooms, target_seconds=target_seconds, bpm=bpm, include_close_ups=False)
    if not line.shots:
        return line

    room_photo_ids = {
        room.room_id: room.photo_ids
        for room in rooms
        if is_detail_eligible(room.room_type)
    }
    selected_room_ids = {shot.room_id for shot in line.shots}
    candidate_photo_ids = {
        photo_id
        for room_id, photo_ids in room_photo_ids.items()
        if room_id in selected_room_ids
        for photo_id in photo_ids
    }
    if not candidate_photo_ids:
        return line

    geometry_rows = db.query(PhotoSceneGeometry).filter(
        PhotoSceneGeometry.photo_id.in_(candidate_photo_ids)
    ).all()
    resolved = resolve_persisted_detail_candidates(
        line.shots,
        room_photo_ids,
        {int(row.photo_id): row.local_metrics or {} for row in geometry_rows},
    )
    line.shots = schedule_resolved_close_ups(line.shots, resolved)
    return line


def latest_timeline(db: Session, job_id: int) -> Timeline | None:
    return (
        db.query(Timeline)
        .filter(Timeline.job_id == job_id)
        .order_by(Timeline.version.desc())
        .first()
    )


def materialize_timeline(db: Session, job_id: int, rooms, *,
                         target_seconds: float, bpm: float,
                         beats_per_shot: int = 2,
                         name: str | None = None) -> Timeline:
    """Write the computed shot line as a new timeline version."""
    line = build_line(db, rooms, target_seconds=target_seconds, bpm=bpm)

    previous = latest_timeline(db, job_id)
    timeline = Timeline(
        job_id=job_id,
        version=(previous.version + 1) if previous else 1,
        status="draft",
        beat_grid={
            "bpm": bpm,
            "beats_per_shot": beats_per_shot,
            "target_seconds": target_seconds,
            "name": name or f"Video {(previous.version + 1) if previous else 1}",
        },
    )
    db.add(timeline)
    # The timeline needs an id before its clips can reference it, and the
    # session does not autoflush here.
    db.flush()

    _write_line_clips(db, timeline, line)
    db.commit()
    db.refresh(timeline)
    return timeline


def populate_timeline(db: Session, timeline: Timeline, rooms, *,
                      target_seconds: float, bpm: float) -> Timeline:
    """Fill an existing blank timeline with the current automatic shot line.

    Population is deliberately separate from materialisation. A later video
    cut starts empty, and filling it must preserve that cut's timeline id so
    Rails' studio record and all editor links remain stable.
    """
    if timeline.timeline_clips:
        raise ValueError("Timeline already contains shots.")

    line = build_line(db, rooms, target_seconds=target_seconds, bpm=bpm)
    _write_line_clips(db, timeline, line)
    db.commit()
    db.refresh(timeline)
    return timeline


def _write_line_clips(db: Session, timeline: Timeline, line) -> None:
    """Persist a computed line into a timeline and recalculate its timing."""
    for index, shot in enumerate(line.shots):
        db.add(TimelineClip(
            timeline_id=timeline.id,
            shot_uid=str(uuid.uuid4()),
            order_index=index,
            room_id=shot.room_id,
            room_type=shot.room_type,
            act=shot.act,
            kind=shot.kind,
            photo_ids=list(shot.photo_ids),
            move_label=shot.move.label,
            move_vector={
                "zoom": shot.move.zoom,
                "pan": shot.move.pan,
                "tilt": shot.move.tilt,
                "roll_deg": shot.move.roll_deg,
            },
            is_close_up=bool(shot.is_close_up),
            crop_box=list(shot.crop_box) if shot.crop_box else None,
            subject_box=list(shot.subject_box) if shot.subject_box else None,
            source_size=list(shot.source_size) if shot.source_size else None,
            detail_subject=shot.detail_subject,
            # The initial edit uses the stable middle of the generated source.
            # The user can still move this window on the timeline.
            trim_start=max(0.0, (source_seconds_for(shot.kind)
                                 - shot.duration_seconds) / 2.0),
            # Hard cuts in all three reference films; no dissolves anywhere.
            transition_type="cut",
        ))

    db.flush()
    timeline.retime()

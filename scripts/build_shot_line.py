"""Build and print the shot line for a listing.

    python -m scripts.build_shot_line <project_id> [--seconds 45] [--bpm 100]
    python -m scripts.build_shot_line <project_id> --json out.json

Rooms come from ground-truth labels when the listing has been labelled, and
from predicted scene components otherwise -- so this works before and after
labelling, and the difference between the two is itself informative.
"""
from __future__ import annotations

import argparse
import collections
import json

from sqlalchemy import text

from app.db.session import SessionLocal
from app.pipeline.phase1_analyze.pairing import persisted_pair_score
from app.pipeline.phase1_analyze.shot_line import (
    RoomCandidate,
    build_shot_line,
    is_persisted_vggt_duplicate,
    room_type_from_name,
    space_scope,
)


def load_rooms(db, project_id: str) -> tuple[list[RoomCandidate], str]:
    job_id = db.execute(text(
        "SELECT max(id) FROM jobs WHERE project_id = :p"), {"p": project_id}).scalar()
    if not job_id:
        raise SystemExit(f"No job for project {project_id}")

    photo_rows = db.execute(text(
        "SELECT id, rails_photo_id, COALESCE(room_override, room_label), COALESCE(final_score,0), position, "
        "COALESCE(is_duplicate, false) "
        "FROM job_photos WHERE job_id = :j"), {"j": job_id}).fetchall()
    # Best verified pair per room, by the planner's own pair score.
    pairs = db.execute(text(
        "SELECT photo_a_id, photo_b_id, debug_metrics FROM photo_relations "
        "WHERE job_id = :j"), {"j": job_id}).fetchall()
    # Older jobs predate the near-zero VGGT duplicate gate. Re-evaluate their
    # persisted numeric evidence without re-running the model, retaining the
    # earliest uploaded frame as the canonical source for the new shot line.
    row_by_id = {row[0]: row for row in photo_rows}
    stale_duplicate_ids: set[int] = set()
    for a, b, dm in pairs:
        if (not is_persisted_vggt_duplicate(dm)
                or int(a) not in row_by_id or int(b) not in row_by_id):
            continue
        left, right = sorted(
            (int(a), int(b)),
            key=lambda photo_id: (int(row_by_id[photo_id][4] or 0), photo_id),
        )
        stale_duplicate_ids.add(right)

    # A VGGT duplicate remains in the scene-map evidence, but it cannot enter
    # the film. Filter both persisted flags and compatibility duplicates before
    # truth or predicted semantic zones are built.
    photos = [row for row in photo_rows
              if not row[5] and row[0] not in stale_duplicate_ids]
    by_key = {r[1]: r for r in photos}
    by_id = {r[0]: r for r in photos}

    pair_score: dict[tuple[int, int], float] = {}
    for a, b, dm in pairs:
        dm = dm or {}
        if dm.get("verification_source") != "vggt_pairwise_v2":
            continue
        score = persisted_pair_score(a, b, dm)
        if score > 0:
            pair_score[(a, b)] = score

    truth = db.execute(text(
        "SELECT room_instances FROM scene_truth_sets WHERE project_id = :p"),
        {"p": project_id}).scalar()

    rooms: list[RoomCandidate] = []
    if truth:
        source = "ground truth"
        for inst in truth:
            ids = [by_key[k][0] for k in inst["photo_keys"] if k in by_key]
            if not ids:
                continue
            labels = [by_id[i][2] for i in ids if by_id[i][2]]
            guess = collections.Counter(labels).most_common(1)[0][0] if labels else "other"
            # The human's instance name outranks the classifier's guess.
            room_type = room_type_from_name(inst["instance"], guess)
            rooms.append(RoomCandidate(
                room_id=inst["instance"], room_type=room_type, photo_ids=ids,
                quality=sum(by_id[i][3] for i in ids) / len(ids),
                photo_positions={i: int(by_id[i][4] or 0) for i in ids},
            ))
    else:
        source = "predicted components"
        component_scene_types = dict(db.execute(text(
            "SELECT id, scene_type FROM scene_components WHERE job_id = :j"),
            {"j": job_id}).fetchall())
        memberships = db.execute(text(
            "SELECT scene_component_id, photo_id FROM scene_component_memberships "
            "WHERE job_id = :j"), {"j": job_id}).fetchall()
        groups = collections.defaultdict(list)
        for cid, pid in memberships:
            groups[cid].append(pid)
        for cid, component_ids in groups.items():
            # VGGT components are continuity structures, not semantic rooms. A
            # living/kitchen/entry component is useful geometry, but collapsing
            # it to its majority label makes the whole component show as a
            # kitchen and destroys the property's story order. Split only the
            # editorial view; all member photos retain their shared geometry.
            zones: dict[str, list[int]] = collections.defaultdict(list)
            for photo_id in component_ids:
                row = by_id.get(photo_id)
                if row is None:
                    continue
                label = (row[2] or "other").strip().lower()
                zones[room_type_from_name(label, label)].append(photo_id)
            for room_type, ids in sorted(zones.items(), key=lambda item: min(by_id[i][4] or 0 for i in item[1])):
                suffix = room_type.replace(" ", "-")
                room_id = f"component-{cid}" if len(zones) == 1 else f"component-{cid}:{suffix}"
                rooms.append(RoomCandidate(
                    room_id=room_id,
                    room_type=room_type,
                    photo_ids=ids,
                    quality=sum(by_id[i][3] for i in ids) / len(ids),
                    photo_positions={i: int(by_id[i][4] or 0) for i in ids},
                    parent_component_id=int(cid),
                    scene_type=component_scene_types.get(cid),
                ))

    for room in rooms:
        found = []
        for a in room.photo_ids:
            for b in room.photo_ids:
                if a >= b:
                    continue
                sc = pair_score.get((a, b)) or pair_score.get((b, a))
                if sc is not None:
                    found.append((a, b, sc))
        room.pairs = sorted(found, key=lambda p: -p[2])
    return rooms, source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_id")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rooms, source = load_rooms(db, args.project_id)
    finally:
        db.close()

    line = build_shot_line(rooms, target_seconds=args.seconds, bpm=args.bpm)
    pos = {}
    print(f"\n  rooms from {source}: {len(rooms)}   target {args.seconds:.0f}s @ {args.bpm:.0f} BPM")
    print(f"  {len(line.shots)} shots x {line.shots[0].duration_seconds:.3f}s = "
          f"{line.duration:.1f}s")
    closeups = sum(1 for s in line.shots if s.is_close_up)
    print(f"  hero: {line.hero_room_id}    "
          f"keyframe pairs: {100*line.keyframe_fraction():.0f}%    "
          f"close-ups: {closeups} ({100*closeups/max(len(line.shots),1):.0f}%)")
    if line.dropped_rooms:
        print(f"  dropped: {', '.join(line.dropped_rooms)}")
    print(f"\n  {'#':>3} {'t':>6} {'act':>4}  {'room':<22}{'source':<16}move")
    for s in line.shots:
        if s.is_close_up:
            src = f"CLOSE-UP of {s.photo_ids[0]}"
        elif s.uses_keyframes:
            src = "PAIR " + "+".join(str(p) for p in s.photo_ids)
        else:
            src = f"photo {s.photo_ids[0]}"
        star = " *" if s.is_hero else "  "
        print(f"  {s.index:>3} {s.start_seconds:>6.2f} {s.act:>4}{star}{s.room_id[:20]:<22}"
              f"{src:<16}{s.move.label}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "bpm": line.bpm, "duration": line.duration,
                "hero_room_id": line.hero_room_id,
                "shots": [{
                    "index": s.index, "room_id": s.room_id, "room_type": s.room_type,
                    "space_scope": space_scope(s.room_type),
                    "photo_ids": s.photo_ids, "kind": s.kind, "act": s.act,
                    "start_seconds": s.start_seconds,
                    "duration_seconds": s.duration_seconds,
                    "is_hero": s.is_hero,
                    "move": {"zoom": s.move.zoom, "pan": s.move.pan,
                             "tilt": s.move.tilt, "roll_deg": s.move.roll_deg,
                             "label": s.move.label},
                } for s in line.shots],
            }, fh, indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()

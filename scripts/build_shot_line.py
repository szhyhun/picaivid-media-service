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
from app.pipeline.phase1_analyze.shot_line import (
    RoomCandidate,
    build_shot_line,
    room_type_from_name,
)


def load_rooms(db, project_id: str) -> tuple[list[RoomCandidate], str]:
    job_id = db.execute(text(
        "SELECT max(id) FROM jobs WHERE project_id = :p"), {"p": project_id}).scalar()
    if not job_id:
        raise SystemExit(f"No job for project {project_id}")

    photos = db.execute(text(
        "SELECT id, rails_photo_id, room_label, COALESCE(final_score,0), position "
        "FROM job_photos WHERE job_id = :j"), {"j": job_id}).fetchall()
    by_key = {r[1]: r for r in photos}
    by_id = {r[0]: r for r in photos}

    # Best verified pair per room, by the planner's own pair score.
    pairs = db.execute(text(
        "SELECT photo_a_id, photo_b_id, debug_metrics FROM photo_relations "
        "WHERE job_id = :j"), {"j": job_id}).fetchall()
    pair_score: dict[tuple[int, int], float] = {}
    for a, b, dm in pairs:
        dm = dm or {}
        if dm.get("direct_membership_edge") or dm.get("same_component"):
            pair_score[(a, b)] = float(dm.get("pair_score") or 0.0)

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
            ))
    else:
        source = "predicted components"
        memberships = db.execute(text(
            "SELECT scene_component_id, photo_id FROM scene_component_memberships "
            "WHERE job_id = :j"), {"j": job_id}).fetchall()
        groups = collections.defaultdict(list)
        for cid, pid in memberships:
            groups[cid].append(pid)
        for cid, ids in groups.items():
            labels = [by_id[i][2] for i in ids if i in by_id and by_id[i][2]]
            room_type = collections.Counter(labels).most_common(1)[0][0] if labels else "other"
            rooms.append(RoomCandidate(
                room_id=f"component-{cid}", room_type=room_type,
                photo_ids=[i for i in ids if i in by_id],
                quality=sum(by_id[i][3] for i in ids if i in by_id) / max(len(ids), 1),
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

"""Score cinematic pairing against the owner's labeled preferred pairs.

Codex review #7: the pairing numbers were produced by ad-hoc scripts, so they were
not reproducible and the 48-label signal was exposed to accidental reinterpretation.
This joins stable truth ids (rails_photo_id) to cached sweep evidence and reports
per-listing coverage and ranking metrics.

    python -m scripts.eval_pairing
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from sqlalchemy import text

from app.db.session import SessionLocal
from app.pipeline.phase1_analyze.pairing import score_pair, select_for_room

SWEEP = os.path.join(os.path.dirname(__file__), "..", "tmp_sweep")


def _load(db, project_id: str, job_id: int):
    """Resolve labels through rails_photo_id, never job_photos.id (reassigned per run)."""
    row = db.execute(
        text("SELECT preferred_pairs, room_instances, listing_slug FROM scene_truth_sets "
             "WHERE project_id = :p AND status = 'complete'"), {"p": project_id}
    ).fetchone()
    if row is None:
        return None
    photos = db.execute(
        text("SELECT id, rails_photo_id, position FROM job_photos WHERE job_id = :j"), {"j": job_id}
    ).fetchall()
    id_by_rails = {r.rails_photo_id: int(r.id) for r in photos}
    position = {int(r.id): r.position for r in photos}
    room = {}
    for entry in row.room_instances or []:
        for rails_id in entry.get("photo_keys", []):
            if rails_id in id_by_rails:
                room[id_by_rails[rails_id]] = entry["instance"]
    preferred = defaultdict(set)
    for a, b in row.preferred_pairs or []:
        if a in id_by_rails and b in id_by_rails:
            left, right = id_by_rails[a], id_by_rails[b]
            if room.get(left) and room.get(left) == room.get(right):
                preferred[room[left]].add((min(left, right), max(left, right)))
    return row.listing_slug, room, preferred, position


def evaluate() -> None:
    db = SessionLocal()
    try:
        rows = []
        totals = defaultdict(int)
        for path in sorted(glob.glob(os.path.join(SWEEP, "*.primary.json"))):
            payload = json.load(open(path))
            loaded = _load(db, payload["project_id"], payload["job_id"])
            if loaded is None:
                continue
            slug, room, preferred, position = loaded

            by_room = defaultdict(list)
            for pair in payload["pairs"]:
                left, right = pair["photo_a_id"], pair["photo_b_id"]
                if room.get(left) and room.get(left) == room.get(right):
                    by_room[room[left]].append(pair)

            rooms_with_labels = any_hit = full_hit = chosen_total = 0
            rank_hits = 0
            for room_name, evidence in by_room.items():
                if room_name not in preferred:
                    continue
                rooms_with_labels += 1
                chosen, _unpaired = select_for_room(evidence, room_name)
                keys = {(min(c.photo_a, c.photo_b), max(c.photo_a, c.photo_b)) for c in chosen}
                chosen_total += len(chosen)
                if keys & preferred[room_name]:
                    any_hit += 1
                if preferred[room_name] <= keys:
                    full_hit += 1
                # ranking: is the owner's pair the single highest-scoring one in the room?
                ranked = sorted(evidence, key=lambda e: -score_pair(e))
                if ranked:
                    top = (min(ranked[0]["photo_a_id"], ranked[0]["photo_b_id"]),
                           max(ranked[0]["photo_a_id"], ranked[0]["photo_b_id"]))
                    if top in preferred[room_name]:
                        rank_hits += 1
            rows.append((slug, rooms_with_labels, any_hit, full_hit, rank_hits, chosen_total))
            totals["rooms"] += rooms_with_labels
            totals["any"] += any_hit
            totals["full"] += full_hit
            totals["rank"] += rank_hits
            totals["pairs"] += chosen_total
    finally:
        db.close()

    if not rows:
        print("No sweep manifests found. Run scripts.sweep_pairs first.")
        return
    print(f"{'listing':<22}{'rooms':>7}{'>=1 chosen':>12}{'all chosen':>12}{'top-1':>8}{'pairs':>7}")
    for slug, rooms, any_hit, full_hit, rank, pairs in rows:
        print(f"{slug[:22]:<22}{rooms:>7}{any_hit:>12}{full_hit:>12}{rank:>8}{pairs:>7}")
    rooms = max(totals["rooms"], 1)
    print(f"\n{'TOTAL':<22}{totals['rooms']:>7}{totals['any']:>12}{totals['full']:>12}"
          f"{totals['rank']:>8}{totals['pairs']:>7}")
    print(f"\ncoverage (room contains >=1 owner-chosen pair): {totals['any']}/{rooms} = {totals['any']/rooms:.0%}")
    print(f"exact     (room contains ALL owner pairs):      {totals['full']}/{rooms} = {totals['full']/rooms:.0%}")
    print(f"top-1     (owner pair is highest scoring):      {totals['rank']}/{rooms} = {totals['rank']/rooms:.0%}")


if __name__ == "__main__":
    evaluate()

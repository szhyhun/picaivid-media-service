"""Score the current pipeline against human ground truth (Scene-Graph V2, Stage 0 §3.4).

Metrics are reported per listing and kept separate by class: physical-room
membership, open-plan continuity, and interpolation safety are different
questions and averaging them hides failures.

Usage:
    python -m tests.scene_graph_eval                 # all completed truth sets
    python -m tests.scene_graph_eval <project_id>    # one listing
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import text

from app.db.session import SessionLocal


@dataclass
class ListingScore:
    listing: str
    photos: int
    truth_rooms: int
    predicted_components: int
    singleton_fraction: float
    membership_precision: float
    membership_recall: float
    space_precision: float
    space_recall: float
    interpolation_precision: float
    cross_room_interpolation_edges: int
    rooms_split: int
    components_contaminated: int
    story_bridges_merged: int


def _fetch(db, project_id: str):
    truth = db.execute(
        text("SELECT room_instances, open_plan_groups, story_bridges, last_job_id, listing_slug "
             "FROM scene_truth_sets WHERE project_id = :p AND status = 'complete'"),
        {"p": project_id},
    ).fetchone()
    if truth is None:
        return None
    job_id = db.execute(
        text("SELECT id FROM jobs WHERE project_id = :p ORDER BY created_at DESC LIMIT 1"),
        {"p": project_id},
    ).scalar()
    if job_id is None:
        return None
    photos = db.execute(
        text("SELECT id, rails_photo_id, position FROM job_photos WHERE job_id = :j"), {"j": job_id}
    ).fetchall()
    memberships = db.execute(
        text("SELECT photo_id, scene_component_id FROM scene_component_memberships WHERE job_id = :j"),
        {"j": job_id},
    ).fetchall()
    relations = db.execute(
        text("SELECT photo_a_id, photo_b_id, continuity_type, is_connected "
             "FROM photo_relations WHERE job_id = :j"),
        {"j": job_id},
    ).fetchall()
    return truth, job_id, photos, memberships, relations


def score(db, project_id: str) -> ListingScore | None:
    fetched = _fetch(db, project_id)
    if fetched is None:
        return None
    truth, job_id, photos, memberships, relations = fetched

    id_by_rails = {row.rails_photo_id: row.id for row in photos}
    keys = set(id_by_rails)

    # --- ground truth: physical rooms, and the looser open-plan spaces ---
    room_of: dict[int, str] = {}
    for entry in truth.room_instances or []:
        for rails_id in entry.get("photo_keys", []):
            if rails_id in id_by_rails:
                room_of[id_by_rails[rails_id]] = entry["instance"]
    space_of = dict(room_of)
    for group in truth.open_plan_groups or []:
        canonical = sorted(group)[0]
        for photo_id, room in list(space_of.items()):
            if room in group:
                space_of[photo_id] = canonical

    photo_ids = sorted(room_of)
    truth_same = {
        (a, b) for a, b in combinations(photo_ids, 2) if room_of[a] == room_of[b]
    }

    # --- predictions ---
    component_of = {row.photo_id: row.scene_component_id for row in memberships}
    predicted_same = {
        (a, b) for a, b in combinations(photo_ids, 2)
        if component_of.get(a) is not None and component_of.get(a) == component_of.get(b)
    }

    def prf(truth_pairs: set) -> tuple[float, float]:
        tp = len(truth_pairs & predicted_same)
        fp = len(predicted_same - truth_pairs)
        fn = len(truth_pairs - predicted_same)
        return (tp / (tp + fp) if (tp + fp) else 1.0,
                tp / (tp + fn) if (tp + fn) else 1.0)

    precision, recall = prf(truth_same)
    # Lenient view: photos in one open-plan space count as together. The product
    # has not decided whether open-plan should merge (plan section 14), so both
    # are reported and neither is treated as the single truth.
    truth_space = {(a, b) for a, b in combinations(photo_ids, 2) if space_of[a] == space_of[b]}
    space_precision, space_recall = prf(truth_space)

    # --- interpolation safety: transition edges must stay inside one room ---
    interp = [
        (r.photo_a_id, r.photo_b_id) for r in relations
        if r.continuity_type == "interpolation_safe"
    ]
    interp_in_truth = [
        p for p in interp
        if p[0] in room_of and p[1] in room_of and room_of[p[0]] == room_of[p[1]]
    ]
    cross = [
        p for p in interp
        if p[0] in room_of and p[1] in room_of and room_of[p[0]] != room_of[p[1]]
    ]
    interp_precision = len(interp_in_truth) / len(interp) if interp else 1.0

    # --- structural failures ---
    comps = {}
    for photo_id, comp in component_of.items():
        if comp is not None and photo_id in room_of:
            comps.setdefault(comp, set()).add(room_of[photo_id])
    contaminated = sum(1 for rooms in comps.values() if len(rooms) > 1)
    rooms_to_comps: dict[str, set] = {}
    for photo_id, room in room_of.items():
        rooms_to_comps.setdefault(room, set()).add(component_of.get(photo_id))
    split = sum(1 for c in rooms_to_comps.values() if len(c) > 1)

    sizes: dict[int, int] = {}
    for comp in component_of.values():
        if comp is not None:
            sizes[comp] = sizes.get(comp, 0) + 1
    singles = sum(1 for n in sizes.values() if n == 1)

    bridges_merged = 0
    for pair in truth.story_bridges or []:
        ids = [id_by_rails.get(k) for k in pair]
        if all(ids) and component_of.get(ids[0]) is not None and component_of.get(ids[0]) == component_of.get(ids[1]):
            bridges_merged += 1

    return ListingScore(
        listing=truth.listing_slug or project_id[:8],
        photos=len(photo_ids),
        truth_rooms=len(truth.room_instances or []),
        predicted_components=len(sizes),
        singleton_fraction=singles / len(sizes) if sizes else 0.0,
        membership_precision=precision,
        membership_recall=recall,
        space_precision=space_precision,
        space_recall=space_recall,
        interpolation_precision=interp_precision,
        cross_room_interpolation_edges=len(cross),
        rooms_split=split,
        components_contaminated=contaminated,
        story_bridges_merged=bridges_merged,
    )


def main() -> None:
    db = SessionLocal()
    try:
        if len(sys.argv) > 1:
            project_ids = sys.argv[1:]
        else:
            project_ids = [
                row.project_id for row in db.execute(
                    text("SELECT project_id FROM scene_truth_sets WHERE status = 'complete' ORDER BY updated_at")
                ).fetchall()
            ]
        results = [r for r in (score(db, pid) for pid in project_ids) if r is not None]
    finally:
        db.close()

    if not results:
        print("No completed ground truth found.")
        return

    header = f"{'listing':<20}{'photos':>7}{'rooms':>6}{'comps':>6}{'single%':>8}{'m-prec':>8}{'m-rec':>7}{'s-prec':>8}{'s-rec':>7}{'interp':>8}{'xroom':>7}{'split':>7}{'contam':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.listing[:20]:<20}{r.photos:>7}{r.truth_rooms:>6}{r.predicted_components:>6}"
              f"{r.singleton_fraction*100:>7.0f}%{r.membership_precision:>8.2f}{r.membership_recall:>7.2f}"
              f"{r.space_precision:>8.2f}{r.space_recall:>7.2f}"
              f"{r.interpolation_precision:>8.2f}{r.cross_room_interpolation_edges:>7}"
              f"{r.rooms_split:>7}{r.components_contaminated:>8}")
    print()
    print("m-prec/m-rec = strict: same physical ROOM INSTANCE")
    print("s-prec/s-rec = lenient: same SPACE (open-plan groups merged)")
    print("xroom  = interpolation edges crossing a room boundary (target 0)")
    print("split  = ground-truth rooms broken across multiple components")
    print("contam = components containing photos from 2+ different rooms")
    print("bridge!= story-bridge pairs wrongly merged into one component (target 0)")


if __name__ == "__main__":
    main()

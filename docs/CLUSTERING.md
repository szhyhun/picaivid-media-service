# VGGT Clustering

## Core idea

Phase 1 no longer chooses clips from pair-matcher scores. It reconstructs geometry first, then derives render groups from that geometry.

## Flow

1. load project photos
2. run VGGT on the project set or overlapping windows
3. persist per-photo geometry
4. measure bidirectional depth overlap, depth consistency, normalized reprojection, and relative pose for candidate pairs
5. classify continuity directly from the jointly reconstructed VGGT geometry and run focused VGGT tracks inside candidate components
6. split scene components only on `interpolation_safe` edges
7. order photos with deterministic beam search over verified edges
8. derive one-to-four-photo render groups and a renderer-neutral ShotPlan
9. choose motion from geometry confidence

## Scene components

Each component stores:

- scene type (`interior`, `exterior`, `drone`, `mixed`)
- ordered photo IDs
- hero/bridge/outlier roles
- geometry confidence
- track coverage
- depth range
- recommended motion affordance

## Indoor / outdoor safeguards

- indoor and outdoor photos do not share one component by default
- cross-domain joins are penalized in relation scoring
- interior-to-exterior joins require strong bridge evidence
- mixed components are split again if cross-domain support is weak
- weakly attached photos are ejected as outliers instead of poisoning a larger scene

This is intentional. The renderer should prefer separate high-confidence room or exterior clips over one bad global scene.

## Relation debug

Two-photo inspection now answers:

- are they in the same component?
- how much overlap do they have?
- how strong is track support?
- is this a bridge edge?
- what is the relative transform?

Relation classes are intentionally strict:

- `duplicate`: very high overlap and near-zero baseline
- `interpolation_safe`: VGGT overlap, depth agreement, reprojection, and rotation all pass
- `doorway_bridge`: editorial continuity only; never used as an interpolation request
- `cut_only` / `unrelated`: safe editorial cut or no use

## Shot plan

`GET /api/projects/:id/shot_plan` returns the ordered renderer-neutral plan. It records the
VGGT checkpoint hash, repository commit, device, precision, inference strategy, story role,
keyframe IDs, transition choice, motion intent, confidence, and any reason a multi-view move
was rejected. `opening`, `hero`, `closing`, and `exclude` are persisted with the source photo
metadata before analysis; explicit roles override automatic ranking.

## Motion guidance

- weak geometry -> `static` or `micro_push_in`
- medium geometry -> `reveal` or `parallax`
- strong geometry -> `multi_view`

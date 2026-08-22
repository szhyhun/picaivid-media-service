# VGGT Clustering

## Core idea

Phase 1 no longer chooses clips from pair-matcher scores. It reconstructs geometry first, then derives render groups from that geometry.

## Flow

1. load project photos
2. run VGGT on the project set or overlapping windows
3. persist per-photo geometry
4. measure confidence-weighted bidirectional surface overlap, depth consistency, normalized reprojection, and relative pose
5. build an adaptive mutual-neighbor graph directly from jointly reconstructed Omega geometry
6. build scene components from `same_scene` and `interpolation_safe` edges
7. order photos with deterministic beam search over verified edges
8. derive one-to-four-photo render groups and a renderer-neutral ShotPlan
9. choose motion from geometry confidence

## Scene components

Each component stores:

- scene type (`interior`, `exterior`, `drone`, `mixed`)
- ordered photo IDs
- hero/bridge/outlier roles
- geometry confidence
- connected-surface coverage
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
- how strong is mutual 3D surface support?
- is this a bridge edge?
- what is the relative transform?

Relation classes are intentionally strict:

- `duplicate`: very high overlap and near-zero baseline
- `interpolation_safe`: same scene with a camera path suitable for continuous interpolation
- `same_scene`: verified views of one physical scene with a wider camera change; keep them in one storyboard shot and use a matched cinematic transition
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

Auto room labels name and story-order a geometry group. They never split a verified Omega scene
component; only explicit editorial overrides may force a separate group.

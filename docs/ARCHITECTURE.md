# Architecture

## Product priority

- VGGT is the star and remains the core scene-understanding system.
- The primary product is cinematic video from listing photos.
- A future `tour_3d` product may reuse denser capture and splat delivery, but it is a secondary branch and should not displace the VGGT-first cinematic pipeline.
- PlayCanvas Gaussian-splat tooling is reference material for that later branch:
  - [playcanvas/supersplat](https://github.com/playcanvas/supersplat)
  - [playcanvas/model-viewer](https://github.com/playcanvas/model-viewer)

## Pipeline shape

1. Rails creates a media job and writes project photos.
2. Media-service phase 1 runs VGGT over the project photo set.
3. VGGT outputs are converted into:
   - `scene_components`
   - `scene_component_memberships`
   - `photo_scene_geometry`
   - `photo_relations`
4. Confidence-weighted Omega surface overlap, depth agreement, and deterministic ordering form a renderer-neutral `ShotPlan`.
5. Render clusters and motion decisions are derived from the verified plan.
6. Phase 2 generates clips from ordered shot-plan inputs.

For any future dense-capture or splat-backed workflow, VGGT should still be treated as the story and scene-logic layer, with 3D reconstruction acting as an execution enhancement rather than a replacement for the core planner.

## Storage model

- Postgres stores metadata and planning state.
- S3 stores dense geometry artifacts:
  - depth maps
  - point maps
  - optional sparse scene exports
  - optional sparse relation evidence

## Scene-first rules

- Omega's shared 3D geometry decides connectivity
- `same_scene` groups views of one physical area; `interpolation_safe` is the stricter camera-motion subset
- room labels are hints, not truth
- indoor and outdoor components use different thresholds
- cross-room parallax is only allowed when actual geometry continuity exists

## Debug surface

- `scenes/debug` explains component-level decisions
- `relations/debug` explains two-photo continuity using derived VGGT relations
- `shot_plan` exposes the ordered cinematic plan, runtime provenance, and per-shot evidence

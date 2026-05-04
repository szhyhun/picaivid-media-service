# Architecture

## Pipeline shape

1. Rails creates a media job and writes project photos.
2. Media-service phase 1 runs VGGT over the project photo set.
3. VGGT outputs are converted into:
   - `scene_components`
   - `scene_component_memberships`
   - `photo_scene_geometry`
   - `photo_relations`
4. Render clusters and motion decisions are derived from scene geometry.
5. Phase 2 generates clips from ordered cluster inputs.

## Storage model

- Postgres stores metadata and planning state.
- S3 stores dense geometry artifacts:
  - depth maps
  - point maps
  - optional sparse scene exports
  - optional track bundles

## Scene-first rules

- geometry decides connectivity
- room labels are hints, not truth
- indoor and outdoor components use different thresholds
- cross-room parallax is only allowed when actual geometry continuity exists

## Debug surface

- `scenes/debug` explains component-level decisions
- `relations/debug` explains two-photo continuity using derived VGGT relations

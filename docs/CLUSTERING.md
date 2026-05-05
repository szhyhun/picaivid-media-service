# VGGT Clustering

## Core idea

Phase 1 no longer chooses clips from pair-matcher scores. It reconstructs geometry first, then derives render groups from that geometry.

## Flow

1. load project photos
2. run VGGT on the project set or overlapping windows
3. persist per-photo geometry
4. derive photo relations from pose, depth, overlap, and track support
5. split the project graph into scene components
6. order photos inside each component
7. derive render clusters
8. choose motion from geometry confidence

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

## Motion guidance

- weak geometry -> `static` or `micro_push_in`
- medium geometry -> `reveal` or `parallax`
- strong geometry -> `multi_view`

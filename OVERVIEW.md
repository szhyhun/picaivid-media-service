# Picaivid Media Pipeline Master Design

This document is the single source of truth for the photo to video generation system. It fully replaces previous guidance. The system is designed for depth aware AI motion, phased processing, database driven state, and future interactive editing.

---

## Goals

- Produce high quality real estate videos with realistic depth and motion.
- Use AI for motion, interpolation, and novel view synthesis when appropriate.
- Persist all pipeline state in Postgres for inspection, overrides, and editing.
- Store only large binary artifacts in S3.
- Support future interactive timeline editing in the UI.
- Allow partial recompute of pipeline phases.
- Run locally on macOS and at scale on AWS ECS GPU.

---

## Non Goals

- No black box monolithic jobs.
- No S3 JSON as system of record.
- No UI coupled rendering logic.
- No restriction on using AI for motion.

---

## High Level Architecture

### Control Plane

- Postgres is the system of record for all plans, clips, timelines, edits, and job state.
- Rails and Media Service both read and write Postgres.

### Data Plane

- S3 stores photos, intermediate clip mp4 files, and final mp4 outputs.

### Compute Plane

- Python media service running on AWS ECS.
- GPU instances for AI rendering.
- Local CPU mode for development with reduced capability.

### Orchestration

- Rails enqueues a single SQS job per video with job_id.
- Media worker executes phased pipeline based on job state in Postgres.
- Worker can be instructed to stop or resume at specific phases.

---

## Single Job With Internal Phases

Rails always sends one job type: job_type = create_video

The worker internally runs phased execution and persists results after each phase.

---

# Core Design Principle

Planner → Generator → Validator

- Planner uses deterministic models to decide what motion is allowed.
- Generator (LTX-2) produces motion.
- Validator re-analyzes output and may downgrade or re-render.

This ensures realism and prevents hallucinated geometry.

---

# Phase 1 — Analyze And Plan

## Purpose

- Understand photo set
- Group by room
- Score quality
- Integrate manual overrides
- Decide motion strategy safely

## Models And Tools

- OpenCLIP for embeddings and clustering
- MiDaS for depth estimation used for planning and validation
- Optional COLMAP for overlap and multi view eligibility validation

## MiDaS Policy

MiDaS is a planning and validation model, not a motion generator.

It is used to:

- Measure depth variance and layering
- Gate allowed camera motion types
- Assign confidence tiers
- Pre validate multi view eligibility
- Detect flat or low depth scenes
- Support downgrade logic when motion is unsafe

## Important Policy

AI is used in this phase for understanding and validation. Depth and overlap inform what motion is allowed. AI is not restricted here.

## Outputs To Postgres

- photos table updated with scores and labels
- room_clusters created
- analysis_results created

## Key Fields

- room clusters
- confidence tier (low, medium, high)
- hero selection
- recommended motion per cluster
- allowed motion types
- 3D eligibility

## Manual Testing For Phase 1

- Verify room grouping
- Verify hero selection
- Verify confidence tiers
- Verify recommended motion
- Verify manual override effects

---

# Phase 2 — Render Clips

## Purpose

- Generate depth aware motion clips
- Generate transitions between views
- Produce reusable clip assets

## Primary Model

- LTX 2 (open source)

## Usage By Confidence Tier

### Low Confidence

- Single image depth aware motion using LTX 2
- Camera control LoRA
- Motions:
  - Micro push in
  - Micro push out
  - Subtle pan

### Medium Confidence

- Two image interpolation using LTX 2 keyframe interpolation
- Reveal transitions between two views
- Limited parallax

### High Confidence

- Multi image synthesis using LTX 2
- Optional COLMAP only to validate eligibility and camera priors
- LTX 2 remains the generator

## Important Policy

AI is always used for motion to ensure depth and parallax. There is no rule forbidding AI motion.

## Depth Validation Policy

Generated clips may be re analyzed with MiDaS to validate depth consistency.

If depth collapses, warping, or planar artifacts are detected, the system must:

- Downgrade to a safer motion profile
- Re render the clip

## Outputs To Postgres And S3

- clips table rows created
- Each clip mp4 uploaded to S3
- s3_uri stored in clips table

## Clip Metadata

- source photos
- motion type
- model used
- confidence tier
- is_3d
- validation metrics

## Manual Testing For Phase 2

- Visual realism
- Depth correctness
- No geometry artifacts
- Downgrade logic correctness

---

# Phase 3 — Timeline And Beat Sync

## Purpose

- Build editable timeline
- Align montage to music beats
- Apply template logic

## Models And Tools

- librosa for beat detection
- Rule based template engine

## Key Concept

Timeline is data, not video.

Timeline is stored in Postgres and is editable by UI. ffmpeg is only a renderer.

## Outputs To Postgres

- timeline row created
- timeline_clips rows created
- beat_grid stored as JSONB

## Timeline Contains

- ordered clips
- in and out trims
- transitions
- beat alignment
- target duration

## Manual Testing For Phase 3

- pacing
- cut density
- beat snap correctness
- room sequencing

---

# Phase 4 — Final Assembly

## Purpose

- Render final mp4

## Tools

- ffmpeg filter complex

## Responsibilities

- concatenate clips
- apply scaling
- normalize fps
- apply transitions
- mix music
- loudness normalization

## Outputs

- final mp4 uploaded to S3
- jobs table updated with final video uri

## Manual Testing For Phase 4

- audio sync
- transitions
- no dropped frames
- correct resolution

---

# Database As Source Of Truth

## jobs

- id
- listing_id
- status
- current_phase
- template_type
- target_length
- music_uri
- bpm
- beat_offset
- enable_beat_sync

## photos

- id
- job_id
- s3_uri
- room_label
- room_override
- exclude
- manual_metadata jsonb
- sharpness
- exposure_score
- composition_score
- base_score
- final_score

## room_clusters

- id
- job_id
- room_type
- confidence_tier
- sfm_eligible
- image_count
- overlap_score
- depth_variance

## analysis_results

- job_id
- room_cluster_id
- hero_photo_id
- recommended_motion
- allowed_motion_types
- recommended_duration
- tier
- model_recommendation
- debug_metrics jsonb

## clips

- id
- job_id
- room_cluster_id
- source_photo_ids
- motion_type
- model_used
- is_3d
- duration
- s3_uri
- validation_score
- status

## timeline

- id
- job_id
- version
- status
- beat_grid jsonb
- total_duration

## timeline_clips

- id
- timeline_id
- clip_id
- order_index
- in_time
- out_time
- transition_type
- audio_policy

## edits

- id
- timeline_id
- user_id
- edit_type
- payload jsonb
- created_at

---

# Manual Overrides

Upstream may provide per photo metadata:

- hero_global
- hero_room
- hero_priority
- preferred_opening
- preferred_closing
- room_override
- exclude
- detail_hint
- notes

---

# Scoring Changes

final_score = base_score + manual_bonus

manual_bonus:

- +0.30 if hero_global
- +0.20 if hero_room
- +0.10 if preferred_opening
- +0.10 if preferred_closing
- +0.05 * hero_priority
- -1.00 if exclude

---

# Quality Gates

Manual flags cannot override minimum sharpness and exposure thresholds.

---

# Opening Selection Priority

1. preferred_opening
2. hero_global
3. drone if template prefers
4. exterior front
5. best interior

---

# Room Start Priority

1. hero_room
2. preferred_opening
3. highest final_score

---

# Closing Selection Priority

1. preferred_closing
2. hero_global
3. exterior or drone
4. best interior

---

# SQS Job Design

Rails sends:

- job_id
- action = run
- optional start_phase

Media worker reads job state from Postgres and executes correct phase.

---

# Partial Recompute

Supported:

- rerun analysis
- rerun clip rendering
- rebuild timeline
- rerender final video

---

# Future Video Editor

UI edits timeline tables, not video files.

UI capabilities:

- reorder clips
- trim in and out
- change transitions
- remove clips
- swap music
- change pacing

Worker reruns only Phase 4 after UI edits.

---

# Why ffmpeg Remains

- deterministic
- scalable
- high quality
- supports complex filters

Frontend trimmers may be used only for preview.

---

# COLMAP Policy

COLMAP is optional and used only to:

- validate multi view eligibility
- estimate overlap
- sanity check camera geometry

COLMAP is not used for depth planning or single image motion.

MiDaS remains the primary depth signal for planning and validation.

---

# GPU and CPU Task Execution Policy

This system separates workloads into CPU and GPU tasks to optimize cost, reliability, and scalability. All GPU resources are treated as ephemeral, on demand, and cost optimized.

## Task Classification

### GPU Tasks
GPU instances are used only for heavy generation and computer vision workloads:

- Video diffusion generation (LTX-2, WAN 2.2)
- Multi-image depth synthesis or 3D reconstruction
- Large-scale image-to-video generation

### CPU Tasks
CPU instances are used for lightweight or orchestration tasks:

- Phase 1 photo analysis:
  - OpenCLIP embeddings
  - MiDaS or ZoeDepth depth estimation (CPU mode)
  - Room clustering and hero frame scoring
- Timeline assembly and sequencing
- Beat detection / beat grid creation
- FFmpeg timeline concatenation and simple audio overlay
- Manual override processing and validation
- Orchestration and job state updates in Postgres

Local developer machines (including Apple Silicon) are supported for CPU tasks, planning, and debugging. They are not expected to run GPU-intensive video generation.

---

## Spot Instance First Policy

GPU workers must run primarily on AWS EC2 Spot instances.

- On-demand GPU instances may be used only as fallback or emergency capacity.
- Spot instances can be interrupted with a two-minute notice.
- Spot instances may occasionally be unavailable in a given Availability Zone.
- GPU workloads must tolerate interruptions with idempotent, restartable tasks.

---

## Job Chunking and Idempotency

GPU workloads must be chunked into small, restartable units:

- Smallest unit: single video clip render
- Each clip task must:
  - Be idempotent
  - Write outputs to S3 immediately
  - Update Postgres status
  - Be safe to retry without duplicate side effects

CPU tasks should also be idempotent but are not typically interrupted.

---

## Spot Interruption Handling

GPU worker containers must:

- Trap SIGTERM
- Flush logs and mark interrupted clips
- Upload partial outputs if possible
- Exit cleanly within two minutes

ECS managed Spot instance draining must be enabled.

---

## Scale From Zero Policy

- GPU capacity should scale from zero when no heavy tasks exist.
- CPU workers can be persistent or auto-scaled lightly based on queue depth.
- In development, GPU instances can be manually started and stopped as needed.

---

## Persistent Spot for Development

- A single persistent Spot GPU instance may be used for dev and testing.
- Stop instance when not actively generating video.
- Preserve EBS volumes for model caches.
- Do not rely on a permanently running GPU instance.

---

## GPU Instance Class Policy

- Default GPU instance: g5.xlarge (NVIDIA A10G, 24GB VRAM)
- Alternative instances must meet minimum VRAM and CUDA support requirements.
- CPU-only instances are sufficient for all lightweight tasks.

---

## Cost Control Requirements

- Visibility into GPU hours consumed
- Per-job GPU time tracking
- Alerts when GPU usage exceeds thresholds
- System must be designed so full daily workload can run on Spot GPU instances efficiently

---

## Development Expectations

Engineers must be able to:

- Run Phase 1 locally on CPU
- Simulate GPU outputs for dev and testing
- Run full GPU generation only on remote GPU workers

---

# Design Principles

- AI is first class for motion and depth
- Safety via validation and downgrade
- Database is the contract
- ffmpeg is renderer, not editor
- Phases are inspectable and repeatable
- GPU is ephemeral and Spot first
- Local dev works on CPU only

This document defines required behavior for implementation.
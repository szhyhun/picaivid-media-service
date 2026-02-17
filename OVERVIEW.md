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

# Phase 2 — Render Clips (LTX-2)

## Purpose

- Generate depth aware motion clips using LTX-2
- Generate transitions between views
- Produce reusable clip assets
- Apply proper prompting constraints for stable geometry

## Primary Models

- LTX-2 (open source) for all standard video generation
- SEVA for premium multi-view geometry (high confidence + SFM eligible)

## Global LTX-2 Prompting Rules

- Always describe camera motion explicitly
- Always describe speed (slow, subtle, cinematic)
- Always constrain geometry in prompts
- Always add negative constraints to prevent warping and hallucination
- Keep clips 2–4 seconds
- Use CFG scale 3–6 (lower = more stable geometry)
- Prefer 720p generation, upscale to 4K later
- Use 30–50 inference steps
- Fix seed for reproducibility

## LTX-2 Parameter Settings

| Parameter | Value |
|-----------|-------|
| Duration | 2–4 seconds |
| Frame rate | 24fps |
| CFG scale | 3–6 |
| Inference steps | 30–50 |
| Seed | Fixed per job |
| Resolution | 720p (upscale after) |

## Usage By Confidence Tier

### Low Confidence (Single Image Ken Burns)

Use when:
- Only one image available
- Low depth variance cluster (< 0.035)
- No geometry inference possible

Allowed motions: static, micro_push_in, micro_push_out, subtle_pan

Master prompt template:
```
Professional real estate interior video.
Slow smooth cinematic [MOTION] camera movement.
Stable tripod motion.
Natural lighting.
Maintain structural consistency.
Preserve wall alignment.
No object movement.
No distortion.
No warping.
No morphing.
No flicker.
Photorealistic.
```

Alternative for pan motion:
```
Slow smooth pan from left to right.
Camera remains level.
No tilt.
No bending walls.
```

### Medium Confidence (Two Image Interpolation)

Use when:
- Same room with 2 views
- Medium depth variance (0.035–0.06)
- Moderate baseline change

Allowed motions: push_in, push_out, pan_left, pan_right, reveal

Prompt template:
```
Smooth cinematic transition between two views of the same room.
Slow forward [MOTION] motion.
Maintain consistent room geometry.
Preserve window and door placement.
Preserve object positions.
Stable lighting.
No morphing.
No warping.
No texture melting.
 
No bending walls.
Photorealistic interior.
```

For hallways:
```
Slow forward camera movement down hallway.
Maintain depth perspective.
No structural distortion.
Stable alignment.
```

### High Confidence (Multi-Frame Parallax)

Use when:
- 3+ overlapping photos
- High depth variance (> 0.06)
- SFM eligible cluster

Allowed motions: dolly_in, dolly_out, orbit, parallax, multi_view

Prompt template:
```
Cinematic slow [MOTION] camera movement inside room.
Subtle parallax effect.
Natural depth separation.
Maintain structural realism.
No hallucinated objects.
No geometry distortion.
Stable lighting.
Preserve original layout.
Photorealistic.
```

Lateral reveal variant:
```
Slow lateral reveal behind furniture.
Subtle rightward camera motion.
Preserve all object positions.
No melting textures.
No wall bending.
```

## Room-Specific Prompt Modifiers

Exterior/Front:
```
Slow approach toward entrance.
Maintain facade geometry.
Preserve landscaping positions.
Natural outdoor lighting.
```

Kitchen:
```
Slow reveal behind counter.
Preserve appliance positions.
Stable countertop geometry.
```

Drone/Aerial:
```
Slow aerial pullback.
Maintain roof geometry.
Stable horizon line.
No ground warping.
```

## SEVA Integration (Premium Tier)

Use SEVA when:
- 3+ photos with high overlap
- Large baseline differences
- Hallways or wide rooms
- Premium tier subscription

SEVA is geometry-driven. Use minimal prompting:
```
Realistic indoor room video.
Natural lighting.
Photorealistic.
```

Primary control is camera trajectory:
- Forward translation
- Orbit arc
- Lateral sweep
- Smooth spline path

SEVA handles:
- Novel view synthesis
- True perspective consistency
- Occlusion correctness

Do NOT rely on LTX-2 for large geometry shifts. Use SEVA for structural correctness.

## Depth Validation Policy

Generated clips may be re-analyzed with MiDaS to validate depth consistency.

If depth collapses, warping, or planar artifacts are detected:
1. Downgrade to safer motion profile
2. Reduce motion magnitude
3. Re-render with stricter constraints

## Video Upscaling

After LTX-2 generation at 720p:
1. Upscale to 4K using temporal-consistent video upscaler
2. Preserve straight lines and geometry
3. Avoid sharpening artifacts
4. Apply slight denoise if needed

Goal: Enhance detail without changing geometry.

## Outputs To Postgres And S3

- clips table rows created
- Each clip mp4 uploaded to S3
- s3_uri stored in clips table
- prompt_used stored for debugging
- cfg_scale, inference_steps stored

## Clip Metadata

- source_photo_ids
- motion_type
- model_used (LTX-2 single/interp/parallax, SEVA)
- prompt_used
- cfg_scale
- inference_steps
- confidence tier
- is_3d
- validation_score
- upscale_status

## Manual Testing For Phase 2

- Visual realism
- Depth correctness
- No geometry artifacts
- No wall bending
- No texture melting
- Downgrade logic correctness
- Prompt effectiveness

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

# Database Ownership Policy

## Rails Database (Read-Only for Python)

Python media service reads from Rails tables but never writes to them.

### Rails-Owned Tables (source data)

- users
- projects
- photos (original uploads, manual metadata)
- templates
- music_tracks

Python reads Rails photos table to get:
- s3_object_key (photo location in S3)
- room_type (manual override if set)
- metadata (manual annotations)
- position, filename, dimensions

## Python Database (Media Service Owns)

All derived and pipeline data is stored in Python's own tables.

### Python-Owned Tables (derived data)

## jobs

- id
- project_id (references Rails project UUID)
- status
- current_phase
- template_type
- target_length
- music_uri
- bpm (detected in Phase 3)
- beat_offset (detected in Phase 3)
- enable_beat_sync

## job_photos

- id
- job_id
- rails_photo_id (references Rails photo UUID)
- s3_uri (copied from Rails for convenience)
- room_label (AI-detected)
- room_override (copied from Rails)
- exclude
- embedding jsonb (computed by OpenCLIP)
- sharpness
- exposure_score
- composition_score
- base_score
- final_score
- depth_variance
- depth_layers

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
- model_recommendation (LTX-2 single/interp/parallax/multi-view, SEVA)
- prompt_template
- cfg_scale
- inference_steps
- debug_metrics jsonb

## clips

- id
- job_id
- room_cluster_id
- source_photo_ids
- motion_type
- model_used (LTX-2 single/interp/parallax/multi-view, SEVA)
- prompt_used
- cfg_scale
- inference_steps
- is_3d
- duration
- s3_uri
- s3_uri_upscaled
- upscale_status
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

# Motion LoRA Training (Future)

## Objective

Reduce reliance on fragile prompting by training deterministic motion styles.

DO NOT train style LoRAs. Train MOTION LoRAs only.

## Dataset Requirements

- 50–300 short real estate clips
- 2–4 seconds each
- Stable camera motion
- Professional walkthrough footage

## Caption Format Examples

```
slow cinematic push in interior living room stable camera no distortion
smooth lateral pan across kitchen stable geometry
forward hallway dolly realistic perspective no warping
subtle orbit movement around room photorealistic stable motion
```

## Separate LoRAs To Train

1. PushIn_LoRA - forward push movements
2. HallwayForward_LoRA - corridor traversal
3. Orbit_LoRA - circular room reveal
4. LateralPan_LoRA - side-to-side panning

## Training Parameters

- LoRA rank: 8–16
- Target modules: attention + motion layers
- Keep LoRA small and modular

## Inference Usage

```
<pushin_lora:1.1>
Professional interior video.
Stable geometry.
Photorealistic.
```

This makes motion behavior deterministic and reduces prompt engineering complexity.

---

# GPU and CPU Task Execution Policy

This system separates workloads into CPU and GPU tasks to optimize cost, reliability, and scalability. All GPU resources are treated as ephemeral, on demand, and cost optimized.

## Global Architecture Principles

1. Never mix heavy GPU inference with lightweight CPU tasks
2. Separate CPU-only services and GPU services
3. Use Spot instances for inference, training, experimental runs
4. Use On-Demand only for orchestrator, DB, critical services
5. Auto-scale GPU workers based on queue depth
6. Keep clips short (2–4 seconds) to control cost

## Task Classification

### GPU Tasks

GPU instances are used only for heavy generation workloads:

- LTX-2 video diffusion generation
- SEVA multi-view geometry synthesis
- Video upscaling (720p → 4K)
- LoRA training (temporary instances)

### CPU Tasks

CPU instances are used for lightweight or orchestration tasks:

- Phase 1 photo analysis:
  - OpenCLIP embeddings
  - MiDaS depth estimation (CPU mode)
  - Room clustering and hero frame scoring
- Timeline assembly and sequencing
- Beat detection / beat grid creation
- FFmpeg timeline concatenation
- Manual override processing
- Orchestration and job state updates

Local developer machines (including Apple Silicon) are supported for CPU tasks, planning, and debugging. They are not expected to run GPU-intensive video generation.

---

## AWS Instance Strategy

### Development Environment

g5.xlarge (1x NVIDIA A10G 24GB VRAM, 4 vCPU, 16GB RAM)

Use for:
- Testing prompts
- Testing interpolation
- Small production batches

Limits:
- Short clips only
- May require FP16
- No heavy LoRA stacking

### Production Baseline

g5.2xlarge (1x A10G 24GB, 8 vCPU, 32GB RAM)

Good balance for:
- 1080p stable generation
- Multi-frame conditioning
- LoRA usage
- 3–5 second clips

### High Performance Alternative

g6e.2xlarge (L4 GPU, 24GB VRAM)

Better efficiency per dollar, lower power draw, good for diffusion workloads.

### Premium Performance (Training/Batch)

p4d.24xlarge (8x A100 40GB)

Use only for:
- Large parallel batch processing
- LoRA training
- Heavy experiments

Cost heavy — not for baseline inference.

### Video Upscaling

g5.xlarge or g6e.xlarge sufficient.

Upscaling is lighter than diffusion. Can even run on g4dn.xlarge (16GB T4) with lighter upscaler.

Separate upscaler workers from LTX workers.

### SEVA Multi-View

g5.2xlarge recommended.

SEVA needs less VRAM than LTX at high resolution. 24GB sufficient.

If running COLMAP + geometry:
- c6i.4xlarge for COLMAP preprocessing (CPU-heavy)
- Then GPU instance for rendering stage

### CPU-Only Services

t3.large or c6i.large for:
- SQS polling
- DB reads
- Metadata processing
- Orchestration

Never waste GPU instances on orchestration.

---

## Spot Instance First Policy

GPU workers must run primarily on AWS EC2 Spot instances.

- On-demand GPU instances may be used only as fallback or emergency capacity
- Spot instances can be interrupted with two-minute notice
- Spot instances may occasionally be unavailable in a given AZ
- GPU workloads must tolerate interruptions with idempotent, restartable tasks

Typical cost: Spot g5.xlarge ~60–70% cheaper than On-Demand.

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

## Scaling Strategy

Estimated workload:
- 30 photos per listing
- 20 listings per day
- ~2–4 min GPU time per video

With 10 g5.xlarge Spot instances:
- 10 videos parallel
- ~150–200 videos per day capacity

Scale rules:
- Queue length > 5 → scale to 10 workers
- Queue length > 20 → scale to 20 workers
- Idle time > 10 minutes → scale down to zero

---

## Scale From Zero Policy

- GPU capacity should scale from zero when no heavy tasks exist
- CPU workers can be persistent or auto-scaled lightly based on queue depth
- In development, GPU instances can be manually started and stopped as needed

---

## Persistent Spot for Development

- A single persistent Spot GPU instance may be used for dev and testing
- Stop instance when not actively generating video
- Preserve EBS volumes for model caches
- Do not rely on a permanently running GPU instance

---

## Cost Control Requirements

- Keep video duration short (2–4s)
- Generate 720p first, upscale later
- Avoid 4K diffusion directly
- Batch jobs per instance
- Use Spot always unless mission critical
- Separate CPU and GPU services
- Visibility into GPU hours consumed
- Per-job GPU time tracking
- Alerts when GPU usage exceeds thresholds

---

## Development Expectations

Engineers must be able to:

- Run Phase 1 locally on CPU
- Simulate GPU outputs for dev and testing
- Run full GPU generation only on remote GPU workers
- Use Spot g5.xlarge for development
- Start instance only when testing, terminate after use

---

# ML Model Deployment Strategy

Models must be pre-baked into Docker images. No runtime downloads.

## Problem

HuggingFace and other model hubs download models on first use. This causes:
- 1-2 minute startup delays per instance
- Unnecessary network traffic
- Failures if hub is unavailable
- Repeated downloads on every new instance

## Solution: Pre-Baked Docker Images

Models are downloaded during Docker build and included in the image.

### Build Process

```bash
# Dockerfile uses multi-stage build:
# 1. Download all models in build stage
# 2. Copy models into final image

docker build -t picaivid-media:latest .
```

### Model Download Script

```bash
python scripts/download_models.py --cache-dir ./ml_models
```

Downloads:
- Intel/dpt-large (MiDaS depth estimation) - ~1.3GB
- OpenCLIP ViT-B-32 - ~350MB

### Environment Variables

Production containers must set:

```bash
HF_HUB_OFFLINE=1          # Prevent any network downloads
MODEL_CACHE_DIR=/app/ml_models
HF_HOME=/app/ml_models/huggingface
TRANSFORMERS_CACHE=/app/ml_models/huggingface
```

### Docker Image Size

Final image size: ~4-5GB (includes models)

This is acceptable. Startup time is more important than image size.

## Alternative: S3 Model Storage

For dynamic model updates without rebuilding images:

1. Upload models to S3:
```bash
./scripts/sync_models_s3.sh upload
```

2. Download during instance bootstrap:
```bash
./scripts/sync_models_s3.sh download
```

Faster than HuggingFace (within AWS network).

## Alternative: EFS Shared Volume

For shared model cache across all instances:

1. Create EFS volume with models
2. Mount at /app/ml_models
3. All instances share the same cache

Pros: No per-instance download
Cons: EFS latency, additional cost

## Recommended Strategy

1. **Development**: Pre-baked Docker images
2. **Production CPU workers**: Pre-baked Docker images
3. **Production GPU workers**: Pre-baked Docker images
4. **Future LoRA updates**: S3 sync for new LoRAs only

## Image Versioning

Tag images with model version:

```
picaivid-media:v1.0-dpt-large-1.3
picaivid-media:v1.0-dpt-large-1.4
```

When models update, build new image version.

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
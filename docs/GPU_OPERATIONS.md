# GPU Operations

This is the no-AI, copy-paste path for starting and stopping the media GPU worker.

## One-time setup per shell

```bash
cd /Users/serhiizhyhun/Desktop/projects/picaivid/picaivid-media-service

export AWS_PROFILE=picaivid-admin
export AWS_REGION=us-west-2
export GPU_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx
```

Replace `GPU_INSTANCE_ID` with the real EC2 instance id of the media worker.

## Start the GPU worker

```bash
./scripts/aws/gpu.sh start
```

What it does:

1. starts the EC2 instance
2. waits until the instance is running
3. best-effort starts `picaivid-media-worker` over SSM

## Check GPU worker status

```bash
./scripts/aws/gpu.sh status
```

This shows:

- instance state
- instance type
- availability zone
- launch time
- public IP
- private IP

## Stop the GPU worker

```bash
./scripts/aws/gpu.sh stop
```

What it does:

1. best-effort stops `picaivid-media-worker` over SSM
2. stops the EC2 instance
3. waits until the instance is fully stopped

## Direct scripts

If you want the lower-level commands, they remain available:

```bash
./scripts/aws/gpu-start.sh
./scripts/aws/gpu-status.sh
./scripts/aws/gpu-stop.sh
```

## Important notes

1. `start` and `stop` assume the GPU instance already exists.
2. These commands do not create or destroy the EC2 instance.
3. If SSM is unhealthy, the EC2 start/stop still works, but the service start/stop is only best-effort.
4. Stopped EC2 instances do not accrue instance runtime charges, but EBS storage still costs money.

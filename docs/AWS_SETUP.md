# AWS Setup Guide for Picaivid Media Service

This guide covers complete AWS setup from scratch for running GPU video generation.

## Prerequisites

- Credit card for AWS billing
- macOS with Homebrew (for AWS CLI)
- Docker installed locally

---

## Step 1: Create AWS Account

1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Enter email, password, account name
4. Select "Personal" account type
5. Add payment method (credit card required)
6. Verify phone number
7. Select "Basic Support" plan (free)
8. Sign in to AWS Console

---

## Step 2: Create IAM User (Security Best Practice)

Don't use root account for daily operations:

1. Go to **IAM** → **Users** → **Create user**
2. User name: `picaivid-admin`
3. Check "Provide user access to AWS Management Console"
4. Select "I want to create an IAM user"
5. Create password
6. Click **Next**
7. Select "Attach policies directly"
8. Attach these policies:
   - `AmazonEC2FullAccess`
   - `AmazonS3FullAccess`
   - `AmazonSQSFullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
9. Click **Create user**
10. Download credentials CSV

### Create Access Keys

1. Go to **IAM** → **Users** → `picaivid-admin`
2. **Security credentials** tab
3. Click **Create access key**
4. Select "Command Line Interface (CLI)"
5. Check acknowledgment, click **Next**
6. Download the `.csv` file (save securely!)

---

## Step 3: Request GPU Instance Quota

**IMPORTANT: New AWS accounts have 0 GPU quota by default!**

1. Go to **Service Quotas** → **AWS services** → **Amazon EC2**
2. Search for `Running On-Demand G and VT instances`
3. Click on it, then **Request quota increase**
4. Enter **4** (for g5.xlarge which has 4 vCPUs)
5. Add reason: "Running ML video generation workloads"
6. Submit request

Also request Spot quota:
1. Search for `All G and VT Spot Instance Requests`
2. Request increase to **4** vCPUs

**Wait 24-48 hours for approval.** AWS reviews GPU quota requests.

---

## Step 4: Install and Configure AWS CLI

```bash
# Install AWS CLI on macOS
brew install awscli

# Configure with your IAM user credentials
aws configure

# Enter when prompted:
# AWS Access Key ID: [from CSV]
# AWS Secret Access Key: [from CSV]
# Default region name: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

---

## Step 5: Create S3 Bucket

```bash
# Create bucket (name must be globally unique)
aws s3 mb s3://picaivid-media-YOUR-UNIQUE-ID --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket picaivid-media-YOUR-UNIQUE-ID \
  --versioning-configuration Status=Enabled
```

---

## Step 6: Create SQS Queue

```bash
# Create standard queue
aws sqs create-queue \
  --queue-name picaivid-jobs \
  --attributes '{
    "VisibilityTimeout": "300",
    "MessageRetentionPeriod": "86400"
  }'

# Get queue URL
aws sqs get-queue-url --queue-name picaivid-jobs
```

Save the queue URL for later.

---

## Step 7: Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name picaivid-media \
  --region us-east-1

# Get repository URI
aws ecr describe-repositories --repository-name picaivid-media
```

Save the repository URI (looks like: `123456789.dkr.ecr.us-east-1.amazonaws.com/picaivid-media`)

---

## Step 8: Create IAM Role for EC2 Instances

```bash
# Create trust policy file
cat > /tmp/ec2-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name picaivid-gpu-worker-role \
  --assume-role-policy-document file:///tmp/ec2-trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name picaivid-gpu-worker-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name picaivid-gpu-worker-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

aws iam attach-role-policy \
  --role-name picaivid-gpu-worker-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name picaivid-gpu-worker-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name picaivid-gpu-worker-profile \
  --role-name picaivid-gpu-worker-role
```

---

## Step 9: Create Security Group

```bash
# Get your public IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Create security group
aws ec2 create-security-group \
  --group-name picaivid-gpu-sg \
  --description "Security group for GPU workers"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names picaivid-gpu-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow SSH from your IP only
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_IP}/32

echo "Security Group ID: $SG_ID"
```

---

## Step 10: Create Key Pair

```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name picaivid-gpu-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/picaivid-gpu-key.pem

# Set permissions
chmod 400 ~/.ssh/picaivid-gpu-key.pem
```

---

## Step 11: Build and Push Docker Image

```bash
# Navigate to project
cd /path/to/picaivid-media-service

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build GPU image
docker build -f Dockerfile.gpu -t picaivid-media:gpu .

# Tag for ECR
docker tag picaivid-media:gpu \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/picaivid-media:gpu-latest

# Push to ECR
docker push \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/picaivid-media:gpu-latest
```

---

## Step 12: Launch GPU Instance

### Find the Deep Learning AMI

```bash
# Get latest Deep Learning AMI (Ubuntu)
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

echo "AMI ID: $AMI_ID"
```

### Launch Spot Instance

```bash
# Create launch specification
cat > /tmp/spot-spec.json << EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "g5.xlarge",
  "KeyName": "picaivid-gpu-key",
  "SecurityGroupIds": ["$SG_ID"],
  "IamInstanceProfile": {
    "Name": "picaivid-gpu-worker-profile"
  },
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "VolumeSize": 100,
        "VolumeType": "gp3"
      }
    }
  ]
}
EOF

# Request Spot instance
aws ec2 request-spot-instances \
  --spot-price "0.60" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file:///tmp/spot-spec.json
```

### Or Launch On-Demand (more expensive but guaranteed)

```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --key-name picaivid-gpu-key \
  --security-group-ids $SG_ID \
  --iam-instance-profile Name=picaivid-gpu-worker-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --count 1
```

---

## Step 13: Connect and Setup Instance

```bash
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].PublicIpAddress' \
  --output text)

# Connect via SSH
ssh -i ~/.ssh/picaivid-gpu-key.pem ubuntu@$INSTANCE_IP

# On the instance, run setup script:
./scripts/aws/setup_gpu_instance.sh
```

---

## Step 14: Start Worker

```bash
# On the GPU instance
./scripts/aws/start_gpu_worker.sh
```

---

## Monitoring and Costs

### Check Running Instances

```bash
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,LaunchTime]' \
  --output table
```

### Stop Instance (to save costs)

```bash
aws ec2 stop-instances --instance-ids i-XXXXX
```

### Terminate Instance (permanent)

```bash
aws ec2 terminate-instances --instance-ids i-XXXXX
```

### Cost Estimates

| Instance | On-Demand | Spot | Per Hour |
|----------|-----------|------|----------|
| g5.xlarge | ~$1.00/hr | ~$0.40/hr | 24GB VRAM |
| g5.2xlarge | ~$1.50/hr | ~$0.60/hr | 24GB VRAM, more CPU |

**Tips to reduce costs:**
- Use Spot instances (60-70% cheaper)
- Stop instances when not in use
- Use 2-second clips during testing
- Generate at 720p, upscale later

---

## Environment Variables for Worker

Set these on the GPU instance:

```bash
export DATABASE_URL="postgresql://user:pass@your-db-host:5432/picaivid"
export AWS_REGION="us-east-1"
export S3_BUCKET="picaivid-media-YOUR-ID"
export SQS_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT/picaivid-jobs"
export WORKER_TYPE="gpu"
export HF_HUB_OFFLINE="1"
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Should show GPU details. If not:
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Docker GPU Access

```bash
# Test GPU in Docker
docker run --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### ECR Login Issues

```bash
# Re-authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

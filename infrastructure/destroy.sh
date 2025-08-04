#!/bin/bash

# Text Summarization MLOps - AWS Infrastructure Destruction Script
# This script safely destroys all AWS resources created by Terraform

set -e

PROJECT_NAME="text-summariser"
AWS_REGION="us-east-1"

echo "🧹 Destroying infrastructure for ${PROJECT_NAME}"
echo "⚠️  This will delete all AWS resources created by Terraform"

# Confirmation prompt
read -p "Are you sure you want to destroy all resources? (yes/no): " confirmation
if [[ $confirmation != "yes" ]]; then
    echo "❌ Destruction cancelled"
    exit 1
fi

cd infrastructure/terraform

# Step 1: Scale down ECS service to 0
echo "📉 Scaling down ECS service..."
CLUSTER_NAME=$(terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
SERVICE_NAME=$(terraform output -raw ecs_service_name 2>/dev/null || echo "")

if [[ -n "$CLUSTER_NAME" && -n "$SERVICE_NAME" ]]; then
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --desired-count 0 \
        --region ${AWS_REGION} || true

    echo "⏳ Waiting for tasks to stop..."
    sleep 30
fi

# Step 2: Delete ECR images
echo "🗑️  Deleting ECR images..."
ECR_REPOSITORY_URI=$(terraform output -raw ecr_repository_url 2>/dev/null || echo "")
if [[ -n "$ECR_REPOSITORY_URI" ]]; then
    aws ecr batch-delete-image \
        --repository-name ${PROJECT_NAME} \
        --image-ids imageTag=latest \
        --region ${AWS_REGION} || true
fi

# Step 3: Destroy Terraform infrastructure
echo "💥 Destroying Terraform infrastructure..."
terraform destroy -auto-approve

echo ""
echo "✅ Infrastructure destroyed successfully!"
echo "💰 All AWS resources have been cleaned up to avoid charges."

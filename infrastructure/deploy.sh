#!/bin/bash

# Text Summarization MLOps - AWS Deployment Script
# This script builds the Docker image, pushes to ECR, and deploys using Terraform

set -e

# Configuration
PROJECT_NAME="text-summariser"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"

echo "🚀 Starting deployment for ${PROJECT_NAME}"
echo "📍 Region: ${AWS_REGION}"
echo "📦 ECR URI: ${ECR_REPOSITORY_URI}"

# Step 1: Initialize Terraform
echo "🏗️  Initializing Terraform..."
cd infrastructure/terraform
terraform init

# Step 2: Plan Terraform deployment
echo "📋 Planning Terraform deployment..."
terraform plan

# Step 3: Apply Terraform (create infrastructure)
echo "🏗️  Creating infrastructure..."
terraform apply -auto-approve

# Get ECR repository URL from Terraform output
ECR_URL=$(terraform output -raw ecr_repository_url)
echo "📦 ECR Repository: ${ECR_URL}"

# Step 4: Build and push Docker image
echo "🐳 Building Docker image..."
cd ../../
docker build -t ${PROJECT_NAME}:latest .

# Step 5: Tag and push to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URL}

echo "🏷️  Tagging image..."
docker tag ${PROJECT_NAME}:latest ${ECR_URL}:latest

echo "⬆️  Pushing image to ECR..."
docker push ${ECR_URL}:latest

# Step 6: Update ECS service
echo "🔄 Updating ECS service..."
cd infrastructure/terraform
CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
SERVICE_NAME=$(terraform output -raw ecs_service_name)

aws ecs update-service \
    --cluster ${CLUSTER_NAME} \
    --service ${SERVICE_NAME} \
    --force-new-deployment \
    --region ${AWS_REGION}

# Step 7: Get deployment URL
LOAD_BALANCER_URL=$(terraform output -raw load_balancer_url)

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Application URL: ${LOAD_BALANCER_URL}"
echo "📊 Health Check: ${LOAD_BALANCER_URL}/health"
echo "📖 API Docs: ${LOAD_BALANCER_URL}/docs"
echo ""
echo "⏳ Note: It may take 2-3 minutes for the service to be fully available."
echo "🔍 Monitor deployment: aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME}"

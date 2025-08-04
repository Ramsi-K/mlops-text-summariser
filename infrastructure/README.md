# Infrastructure as Code - AWS Deployment

This directory contains Terraform configurations and deployment scripts for deploying the Text Summarization MLOps application to AWS.

## Architecture

The infrastructure includes:

- **Amazon ECS Fargate**: Serverless container hosting
- **Application Load Balancer**: High availability and traffic distribution
- **Amazon ECR**: Docker image registry
- **VPC with Public Subnets**: Secure networking
- **CloudWatch**: Logging and monitoring
- **IAM Roles**: Secure service permissions

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
   ```bash
   aws configure
   ```

2. **Terraform** installed (>= 1.0)
   ```bash
   # On macOS
   brew install terraform

   # On Ubuntu/Debian
   wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
   sudo apt update && sudo apt install terraform
   ```

3. **Docker** installed and running

4. **AWS Account** with sufficient permissions for:
   - ECS, ECR, VPC, ALB, IAM, CloudWatch

## Quick Deployment

### Option 1: Automated Deployment Script

```bash
# Deploy everything with one command
./infrastructure/deploy.sh
```

This script will:
1. Initialize and apply Terraform configuration
2. Build and push Docker image to ECR
3. Deploy application to ECS Fargate
4. Provide the application URL

### Option 2: Manual Step-by-Step

1. **Initialize Terraform**
   ```bash
   cd infrastructure/terraform
   terraform init
   ```

2. **Review and Apply Infrastructure**
   ```bash
   terraform plan
   terraform apply
   ```

3. **Build and Push Docker Image**
   ```bash
   # Get ECR repository URL
   ECR_URL=$(terraform output -raw ecr_repository_url)

   # Build image
   docker build -t text-summariser:latest ../../

   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URL

   # Tag and push
   docker tag text-summariser:latest $ECR_URL:latest
   docker push $ECR_URL:latest
   ```

4. **Update ECS Service**
   ```bash
   aws ecs update-service \
     --cluster $(terraform output -raw ecs_cluster_name) \
     --service $(terraform output -raw ecs_service_name) \
     --force-new-deployment
   ```

## Configuration

### Terraform Variables

Copy and customize the variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

Key variables:
- `aws_region`: AWS region (default: us-east-1)
- `project_name`: Project name (default: text-summariser)
- `fargate_cpu`: CPU units (default: 1024 = 1 vCPU)
- `fargate_memory`: Memory in MiB (default: 2048 = 2GB)
- `app_count`: Number of container instances (default: 2)

### Cost Estimation

**Monthly costs (us-east-1, 2 instances running 24/7):**
- ECS Fargate (1 vCPU, 2GB): ~$30-35/month
- Application Load Balancer: ~$16/month
- ECR storage: ~$1/month (minimal)
- CloudWatch logs: ~$1/month (7-day retention)
- **Total: ~$48-53/month**

**Cost optimization:**
- Use smaller instance sizes for development
- Enable ECS auto-scaling
- Use Spot instances for non-production

## Application URLs

After deployment, access your application:

```bash
# Get the load balancer URL
terraform output load_balancer_url
```

- **Application**: http://your-alb-dns-name/
- **Health Check**: http://your-alb-dns-name/health
- **API Documentation**: http://your-alb-dns-name/docs
- **Prediction Endpoint**: http://your-alb-dns-name/predict

## Monitoring

### CloudWatch Logs
```bash
# View application logs
aws logs tail /ecs/text-summariser --follow
```

### ECS Service Status
```bash
aws ecs describe-services \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --services $(terraform output -raw ecs_service_name)
```

### Load Balancer Health
Check target group health in AWS Console or:
```bash
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups \
    --names text-summariser-tg --query 'TargetGroups[0].TargetGroupArn' --output text)
```

## Updating the Application

1. **Code Changes**: Make changes to your application
2. **Rebuild and Push**:
   ```bash
   docker build -t text-summariser:latest .
   docker tag text-summariser:latest $ECR_URL:latest
   docker push $ECR_URL:latest
   ```
3. **Deploy Update**:
   ```bash
   aws ecs update-service \
     --cluster $(terraform output -raw ecs_cluster_name) \
     --service $(terraform output -raw ecs_service_name) \
     --force-new-deployment
   ```

## Cleanup

### Complete Infrastructure Destruction

```bash
# Automated cleanup
./infrastructure/destroy.sh
```

Or manually:
```bash
cd infrastructure/terraform
terraform destroy
```

**⚠️ Important**: This will delete all resources and cannot be undone.

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check CloudWatch logs: `aws logs tail /ecs/text-summariser --follow`
   - Verify container health check endpoint: `/health`

2. **Load balancer returns 503**
   - ECS tasks may be starting up (wait 2-3 minutes)
   - Check target group health in AWS Console

3. **Terraform apply fails**
   - Ensure AWS credentials are configured
   - Check AWS service limits (VPC, ECS)
   - Verify region availability

4. **Docker push fails**
   - Ensure ECR repository exists: `terraform apply` first
   - Re-authenticate: `aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL`

### Useful Commands

```bash
# Check ECS service events
aws ecs describe-services --cluster CLUSTER_NAME --services SERVICE_NAME --query 'services[0].events[0:5]'

# Check container logs
aws logs get-log-events --log-group-name /ecs/text-summariser --log-stream-name STREAM_NAME

# List running tasks
aws ecs list-tasks --cluster CLUSTER_NAME --service-name SERVICE_NAME
```

## Security Considerations

- Application Load Balancer is internet-facing (port 80 only)
- ECS tasks run in public subnets with security groups
- ECR images are scanned for vulnerabilities
- IAM roles follow least-privilege principle
- CloudWatch logs have 7-day retention (configurable)

For production deployments, consider:
- HTTPS/TLS termination at ALB
- Private subnets for ECS tasks
- WAF for additional protection
- Secrets Manager for sensitive configuration

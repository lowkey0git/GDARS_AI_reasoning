#!/usr/bin/env bash
# Creates the S3 bucket and DynamoDB table needed for Terraform remote state.
# Run once before: cd terraform && terraform init
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET="gdars-terraform-state"
TABLE="gdars-terraform-locks"

echo "==> Bootstrapping Terraform backend in: $REGION"

# S3 bucket
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "    S3 '$BUCKET' already exists"
else
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption \
    --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  echo "    S3 bucket created"
fi

# DynamoDB table
if aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" 2>/dev/null; then
  echo "    DynamoDB '$TABLE' already exists"
else
  aws dynamodb create-table \
    --table-name "$TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$REGION"
  echo "    DynamoDB table created"
fi

echo ""
echo "==> Done. Now run:"
echo "    cd terraform"
echo "    cp terraform.tfvars.example terraform.tfvars"
echo "    # fill in your API keys"
echo "    terraform init"
echo "    terraform apply -var-file=terraform.tfvars"

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket         = "gdars-terraform-state"
    key            = "gdars/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "gdars-terraform-locks"
  }
}

provider "aws" { region = var.aws_region }

data "aws_availability_zones" "available" { state = "available" }
data "aws_caller_identity" "current" {}

locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  name    = "${var.project_name}-vpc"
  cidr    = "10.0.0.0/16"
  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.10.0/24", "10.0.11.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  tags = local.tags
}

# ── ECR ───────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "gdars" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "gdars" {
  repository = aws_ecr_repository.gdars.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "gdars" {
  name                    = "${var.project_name}/env"
  recovery_window_in_days = 7
  tags                    = local.tags
}

resource "aws_secretsmanager_secret_version" "gdars" {
  secret_id     = aws_secretsmanager_secret.gdars.id
  secret_string = jsonencode({
    GROQ_API_KEY          = var.groq_api_key
    RETRIEVAL_TAVILY_API_KEY = var.tavily_api_key
  })
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name   = "${var.project_name}-alb-sg"
  vpc_id = module.vpc.vpc_id
  ingress { from_port = 80;   to_port = 80;   protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }
  egress  { from_port = 0;    to_port = 0;    protocol = "-1";  cidr_blocks = ["0.0.0.0/0"] }
  tags = local.tags
}

resource "aws_security_group" "ecs" {
  name   = "${var.project_name}-ecs-sg"
  vpc_id = module.vpc.vpc_id
  ingress { from_port = 8000; to_port = 8000; protocol = "tcp"; security_groups = [aws_security_group.alb.id] }
  egress  { from_port = 0;    to_port = 0;    protocol = "-1";  cidr_blocks = ["0.0.0.0/0"] }
  tags = local.tags
}

resource "aws_security_group" "redis" {
  name   = "${var.project_name}-redis-sg"
  vpc_id = module.vpc.vpc_id
  ingress { from_port = 6379; to_port = 6379; protocol = "tcp"; security_groups = [aws_security_group.ecs.id] }
  tags = local.tags
}

# ── ElastiCache Redis (working memory Layer 1) ────────────────────────────────
resource "aws_elasticache_subnet_group" "gdars" {
  name       = "${var.project_name}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-redis"
  engine               = "redis"
  node_type            = "cache.t4g.micro"   # cheapest option
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.gdars.name
  security_group_ids   = [aws_security_group.redis.id]
  tags                 = local.tags
}

# ── EFS (persistent storage for ChromaDB + FAISS) ────────────────────────────
resource "aws_efs_file_system" "gdars" {
  creation_token = "${var.project_name}-data"
  encrypted      = true
  tags           = local.tags
}

resource "aws_efs_mount_target" "gdars" {
  for_each        = toset(module.vpc.private_subnets)
  file_system_id  = aws_efs_file_system.gdars.id
  subnet_id       = each.value
  security_groups = [aws_security_group.ecs.id]
}

# ── IAM ───────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow",
                   Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_secrets" {
  name = "read-secrets"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["secretsmanager:GetSecretValue"],
                   Resource = [aws_secretsmanager_secret.gdars.arn] }]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow",
                   Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
  tags = local.tags
}

# CI/CD deploy user (least-privilege)
resource "aws_iam_user" "cicd" {
  name = "${var.project_name}-cicd"
  tags = local.tags
}

resource "aws_iam_access_key" "cicd" {
  user = aws_iam_user.cicd.name
}

resource "aws_iam_user_policy" "cicd" {
  name = "${var.project_name}-cicd-policy"
  user = aws_iam_user.cicd.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid = "ECRAuth",    Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = ["*"] },
      { Sid = "ECRPush",    Effect = "Allow", Resource = [aws_ecr_repository.gdars.arn],
        Action = ["ecr:BatchCheckLayerAvailability","ecr:CompleteLayerUpload","ecr:InitiateLayerUpload",
                  "ecr:PutImage","ecr:UploadLayerPart","ecr:BatchGetImage","ecr:DescribeRepositories"] },
      { Sid = "ECSRead",    Effect = "Allow", Resource = ["*"],
        Action = ["ecs:DescribeTaskDefinition","ecs:DescribeServices","ecs:ListTaskDefinitions"] },
      { Sid = "ECSDeploy",  Effect = "Allow", Resource = ["*"],
        Action = ["ecs:RegisterTaskDefinition","ecs:UpdateService","ecs:DeregisterTaskDefinition"] },
      { Sid = "PassRole",   Effect = "Allow", Action = ["iam:PassRole"],
        Resource = [aws_iam_role.ecs_execution.arn, aws_iam_role.ecs_task.arn] },
      { Sid = "ALBRead",    Effect = "Allow", Action = ["elasticloadbalancing:DescribeLoadBalancers"], Resource = ["*"] },
    ]
  })
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "gdars" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  tags               = local.tags
}

resource "aws_lb_target_group" "gdars" {
  name        = "${var.project_name}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"
  health_check {
    path                = "/ready"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }
  tags = local.tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.gdars.arn
  port              = 80
  protocol          = "HTTP"
  default_action { type = "forward"; target_group_arn = aws_lb_target_group.gdars.arn }
}

# ── CloudWatch ────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "gdars" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14
  tags              = local.tags
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "gdars" {
  name = "${var.project_name}-cluster"
  setting { name = "containerInsights"; value = "enabled" }
  tags = local.tags
}

resource "aws_ecs_cluster_capacity_providers" "gdars" {
  cluster_name       = aws_ecs_cluster.gdars.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight = 1; base = 1
  }
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "gdars" {
  family                   = "${var.project_name}-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  volume {
    name = "gdars-data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.gdars.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([{
    name      = "gdars-api"
    image     = "${aws_ecr_repository.gdars.repository_url}:latest"
    essential = true
    portMappings = [{ containerPort = 8000; hostPort = 8000; protocol = "tcp" }]
    environment = [
      { name = "ENVIRONMENT",                    value = "production" },
      { name = "API_HOST",                       value = "0.0.0.0" },
      { name = "API_PORT",                       value = "8000" },
      { name = "MEMORY_REDIS_URL",               value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0" },
      { name = "MEMORY_CHROMA_PERSIST_DIR",      value = "/app/data/chroma" },
      { name = "MEMORY_FAISS_INDEX_PATH",        value = "/app/data/faiss_index" },
    ]
    secrets = [
      { name = "GROQ_API_KEY",              valueFrom = "${aws_secretsmanager_secret.gdars.arn}:GROQ_API_KEY::" },
      { name = "RETRIEVAL_TAVILY_API_KEY",  valueFrom = "${aws_secretsmanager_secret.gdars.arn}:RETRIEVAL_TAVILY_API_KEY::" },
    ]
    mountPoints = [{ sourceVolume = "gdars-data"; containerPath = "/app/data"; readOnly = false }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.gdars.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/ready || exit 1"]
      interval    = 30; timeout = 10; retries = 3; startPeriod = 90
    }
  }])

  tags = local.tags
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "gdars" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.gdars.id
  task_definition = aws_ecs_task_definition.gdars.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.gdars.arn
    container_name   = "gdars-api"
    container_port   = 8000
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  deployment_circuit_breaker { enable = true; rollback = true }

  depends_on = [aws_lb_listener.http]
  tags       = local.tags

  lifecycle { ignore_changes = [task_definition] }
}

# ── Auto Scaling ──────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "gdars" {
  max_capacity       = var.max_capacity
  min_capacity       = var.desired_count
  resource_id        = "service/${aws_ecs_cluster.gdars.name}/${aws_ecs_service.gdars.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.project_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.gdars.resource_id
  scalable_dimension = aws_appautoscaling_target.gdars.scalable_dimension
  service_namespace  = aws_appautoscaling_target.gdars.service_namespace
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

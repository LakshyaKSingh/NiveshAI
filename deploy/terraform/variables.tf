variable "aws_region" {
  description = "AWS deployment region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (production, staging, dev)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name tag prefix"
  type        = string
  default     = "tradepilot"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t4g.micro"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "tradepilot_db"
}

variable "db_username" {
  description = "PostgreSQL administrator username"
  type        = string
  default     = "tradepilot_admin"
}

variable "api_cpu" {
  description = "ECS Fargate CPU units for API"
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "ECS Fargate Memory (MB) for API"
  type        = number
  default     = 1024
}

variable "worker_cpu" {
  description = "ECS Fargate CPU units for Worker"
  type        = number
  default     = 512
}

variable "worker_memory" {
  description = "ECS Fargate Memory (MB) for Worker"
  type        = number
  default     = 1024
}

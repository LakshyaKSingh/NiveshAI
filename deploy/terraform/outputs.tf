output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer"
  value       = aws_lb.api.dns_name
}

output "rds_endpoint" {
  description = "RDS PostgreSQL database endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "s3_bucket_name" {
  description = "S3 Artifacts bucket name"
  value       = aws_s3_bucket.artifacts.id
}

output "ecs_cluster_name" {
  description = "ECS Cluster Name"
  value       = aws_ecs_cluster.main.name
}

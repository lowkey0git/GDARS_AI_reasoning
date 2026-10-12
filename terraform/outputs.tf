output "alb_url"                { value = "http://${aws_lb.gdars.dns_name}" }
output "ecr_repository_url"    { value = aws_ecr_repository.gdars.repository_url }
output "redis_endpoint"        { value = aws_elasticache_cluster.redis.cache_nodes[0].address }
output "ecs_cluster_name"      { value = aws_ecs_cluster.gdars.name }
output "ecs_service_name"      { value = aws_ecs_service.gdars.name }
output "cloudwatch_log_group"  { value = aws_cloudwatch_log_group.gdars.name }
output "cicd_access_key_id"    { value = aws_iam_access_key.cicd.id }
output "cicd_secret_access_key" {
  value     = aws_iam_access_key.cicd.secret
  sensitive = true
}

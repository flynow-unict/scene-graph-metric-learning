output "website_url" {
  description = "Il link pubblico per accedere al Frontend del progetto"
  value       = "http://${aws_lb.main.dns_name}"
}

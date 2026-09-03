# SSH Key Pair for Ansible access
resource "tls_private_key" "k3s_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}


# AWS Key Pair
resource "aws_key_pair" "k3s_key" {
  key_name   = "${var.project_name}-key"
  public_key = tls_private_key.k3s_key.public_key_openssh
}

# Local file for private key
resource "local_file" "private_key" {
  content         = tls_private_key.k3s_key.private_key_pem
  filename        = "${path.module}/k3s_key.pem"
  file_permission = "0400"
}

data "aws_ami" "ubuntu" {
    most_recent = true
    owners      = ["099720109477"]

    filter {
        name = "name"
        values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
    }
}

# K3s Master Node
resource "aws_instance" "k3s_master" {
    ami = data.aws_ami.ubuntu.id
    instance_type = "t3.micro"
    subnet_id = aws_subnet.public[0].id
    vpc_security_group_ids = [aws_security_group.k3s_nodes.id]
    key_name = aws_key_pair.k3s_key.key_name

    tags = {
        Name = "${var.project_name}-master"
        Role = "master"
    }
}


#K3s Worker Node 1 (Frontend/Backend)
resource "aws_instance" "k3s_worker_1"{
    ami = data.aws_ami.ubuntu.id
    instance_type = "t3.micro"
    subnet_id = aws_subnet.private[0].id
    vpc_security_group_ids = [aws_security_group.k3s_nodes.id]
    key_name = aws_key_pair.k3s_key.key_name

    tags = {
        Name = "${var.project_name}-worker-1"
    }
}


#K3s Worker Node 2 (AI Worker)
resource "aws_instance" "k3s_worker_2" {
    ami = data.aws_ami.ubuntu.id
    instance_type = "t3.micro"
    subnet_id = aws_subnet.private[1].id
    vpc_security_group_ids = [aws_security_group.k3s_nodes.id]
    key_name = aws_key_pair.k3s_key.key_name

    tags = {
        Name = "${var.project_name}-worker-2"
    }
}

variable "LAMBDALABS_API_KEY" {
  type = string
}

variable gpu_type {
    type = string
}

variable lambda_region {
    type = string
}

variable lambda_key_name {
    type = string
}

variable lambda_private_key {
    type = string
}

terraform {
  required_providers {
    lambdalabs = {
      source = "elct9620/lambdalabs"
    }
  }
}

provider "lambdalabs" {
  api_key = var.LAMBDALABS_API_KEY
}

resource "lambdalabs_instance" "training" {
  region_name        = var.lambda_region
  instance_type_name = var.gpu_type

  # Suggest creating a key instead of using resource to make it reusable
  ssh_key_names = [
    var.lambda_key_name
  ]

  connection {
    type     = "ssh"
      user     = "ubuntu"
      private_key = file(var.lambda_private_key)
      host     = self.ip
  }
}

output "instance_ip_addr" {
  value = lambdalabs_instance.training.ip
}
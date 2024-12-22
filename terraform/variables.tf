variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = "ml-training-cluster"
}

variable "cluster_machine_type" {
  description = "Machine type for the cluster nodes"
  type        = string
  default     = "n1-standard-8"
}

variable "cluster_num_nodes" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 3
}

variable "addons" {
  description = "Addons to enable on the cluster"
  type        = list(string)
  default     = ["HorizontalPodAutoscaling", "HttpLoadBalancing"]
}

variable "node_pool_name" {
  description = "Name of the GPU node pool"
  type        = string
  default     = "gpu-pool"
}

variable "node_pool_machine_type" {
  description = "Machine type for the node pool"
  type        = string
  default     = "g2-standard-4"
}

variable "node_pool_accelerators" {
  description = "Accelerators for the node pool"
  type        = map(any)
  default     = {
    type  = "nvidia-l4"
    count = 1
  }
}

variable "node_pool_num_nodes" {
  description = "Initial number of nodes in the node pool"
  type        = number
  default     = 1
}

variable "node_pool_min_nodes" {
  description = "Minimum number of nodes for autoscaling"
  type        = number
  default     = 0
}

variable "node_pool_max_nodes" {
  description = "Maximum number of nodes for autoscaling"
  type        = number
  default     = 3
}


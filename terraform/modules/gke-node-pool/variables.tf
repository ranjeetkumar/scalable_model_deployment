variable "node_pool_name" {
  description = "Name of the GPU node pool"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "location" {
  description = "The location of the node pool"
  type        = string
}

variable "machine_type" {
  description = "Machine type for the node pool"
  type        = string
}

variable "accelerators" {
  description = "Accelerators for the node pool"
  type        = map(any)
}

variable "num_nodes" {
  description = "Initial number of nodes in the node pool"
  type        = number
}

variable "min_nodes" {
  description = "Minimum number of nodes for autoscaling"
  type        = number
}

variable "max_nodes" {
  description = "Maximum number of nodes for autoscaling"
  type        = number
}

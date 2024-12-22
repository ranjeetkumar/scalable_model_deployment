variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
}

variable "machine_type" {
  description = "Machine type for the cluster nodes"
  type        = string
}

variable "num_nodes" {
  description = "Number of nodes in the cluster"
  type        = number
}

variable "location" {
  description = "The location of the cluster"
  type        = string
}

variable "addons" {
  description = "Addons to enable on the cluster"
  type        = list(string)
}

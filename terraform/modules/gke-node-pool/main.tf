resource "google_container_node_pool" "gpu_pool" {
  name       = var.node_pool_name
  cluster    = var.cluster_name
  location   = var.location
  node_count = var.num_nodes

  node_config {
    machine_type = var.machine_type
    guest_accelerator {
      type  = var.accelerators["type"]
      count = var.accelerators["count"]
    }

    
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    metadata = {
      disable-legacy-endpoints = "true"
    }
    tags = ["gpu-node"]
    labels = {
      env = "gpu"
    }
  }

  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }
}

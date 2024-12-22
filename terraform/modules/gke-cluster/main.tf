resource "google_container_cluster" "gke_cluster" {
  name               = var.cluster_name
  location           = var.location
  initial_node_count = var.num_nodes
  deletion_protection = false

  node_config {
    machine_type = var.machine_type
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = contains(var.addons, "HorizontalPodAutoscaling") ? false : true
    }
    http_load_balancing {
      disabled = contains(var.addons, "HttpLoadBalancing") ? false : true
    }
  }
}

output "cluster_name" {
  value = google_container_cluster.gke_cluster.name
}

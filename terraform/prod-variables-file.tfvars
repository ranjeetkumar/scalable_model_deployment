project_id           = "perfect-science-437706-m0"
region               = "us-central1"
zone                 = "us-central1-a"
cluster_name         = "ml-training-cluster"
cluster_machine_type = "n1-standard-8"
cluster_num_nodes    = 3
addons               = ["HorizontalPodAutoscaling", "HttpLoadBalancing"]
node_pool_name       = "gpu-pool"
node_pool_machine_type = "g2-standard-4"
node_pool_accelerators = {
  type  = "nvidia-l4"
  count = 1
}
node_pool_num_nodes  = 1
node_pool_min_nodes  = 0
node_pool_max_nodes  = 3

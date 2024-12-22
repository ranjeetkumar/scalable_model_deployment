provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

module "gke_cluster" {
  source            = "./modules/gke-cluster"
  cluster_name      = var.cluster_name
  machine_type      = var.cluster_machine_type
  num_nodes         = var.cluster_num_nodes
  location          = var.zone
  addons            = var.addons
}

module "gpu_node_pool" {
  source        = "./modules/gke-node-pool"
  cluster_name  = module.gke_cluster.cluster_name
  location      = var.zone
  node_pool_name = var.node_pool_name
  machine_type  = var.node_pool_machine_type
  accelerators  = var.node_pool_accelerators
  num_nodes     = var.node_pool_num_nodes
  min_nodes     = var.node_pool_min_nodes
  max_nodes     = var.node_pool_max_nodes
}


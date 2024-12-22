# Scalable Model Deployment

This project demonstrates an end-to-end machine learning model deployment pipeline using FastAPI, Kubernetes, and Terraform.
## Project structure
```bash
├── api_call.py          # API testing script
├── deployment/          # Deployment configuration
│   ├── Dockerfile      # Container definition
│   ├── main.py        # FastAPI application
│   └── requirements.txt
├── k8s/                # Kubernetes manifests
│   ├── deployment.yaml
│   ├── hpa.yaml       # Horizontal Pod Autoscaler
│   └── service.yaml
├── terraform/          # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── training/           # ML model training
    ├── train.py
    ├── config.yaml
    └── requirements.txt
```
## Prediction Table Schema

The following table defines the schema for the `prediction` table used for storing prediction outputs:

| **Field Name**           | **Type**     | **Mode**    | **Key** | **Collation** | **Default Value** | **Policy Tags** | **Description** |
|---------------------------|--------------|-------------|---------|---------------|-------------------|-----------------|-----------------|
| `prediction_id`           | STRING       | REQUIRED    | -       | -             | -                 | -               | Unique identifier for each prediction |
| `model_version`           | STRING       | REQUIRED    | -       | -             | -                 | -               | Model version used for prediction     |
| `input_shape`             | STRING       | REQUIRED    | -       | -             | -                 | -               | Input data shape                      |
| `predicted_class`         | INTEGER      | REQUIRED    | -       | -             | -                 | -               | Predicted class label                 |
| `class_probabilities`     | FLOAT        | REPEATED    | -       | -             | -                 | -               | Probabilities for each class          |
| `created_at`              | TIMESTAMP    | REQUIRED    | -       | -             | -                 | -               | Timestamp of the prediction           |
| `processing_time_ms`      | FLOAT        | REQUIRED    | -       | -             | -                 | -               | Time taken for prediction (ms)        |

The above table must be present in bigquery table
## Deployment Steps

Follow the steps below to deploy the trained model on **GKE**:
### 1. Create ConfigMap
Configure necessary environment variables for the deployment:
```bash
kubectl create configmap ml-model-config \
  --from-literal=gcp_bucket_name=model_weights_demo \
  --from-literal=bigquery_dataset=model_prediction \
  --from-literal=TRUE
```
### 2. Create Secret
Provide the GCP service account key as a Kubernetes secret:
```bash
kubectl create secret generic gcp-service-account-key \
  --from-file=key.json=/Users/rankum/Downloads/perfect-science-437706-m0-dcc2dbaf6c93.json
```
### 3. Apply Kubernetes Deployment
Deploy the model using the provided Kubernetes deployment YAML file:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/service.yaml
```
## Infrastructure Management with Terraform
The infrastructure for this project is managed using Terraform. Follow these steps to set up and manage the infrastructure:

Initialize Terraform
terraform init
Plan the Infrastructure
terraform plan -var-file="prod-variables-file.tfvars" "-lock=false"
Apply Changes
terraform apply -var-file="prod-variables-file.tfvars" "-lock=false"
Destroy Resources
If needed, destroy the resources:
terraform destroy -var-file="prod-variables-file.tfvars" "-lock=false"





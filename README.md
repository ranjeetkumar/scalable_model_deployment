# Scalable Model Deployment

This project demonstrates the end-to-end process of training and deploying a scalable machine learning model using **TensorFlow** and **Keras**. The system includes:

- **Data Preprocessing**: Preparing the dataset for training.
- **Model Training**: Using TensorFlow/Keras for model creation and training.
- **Model Checkpointing**: Saving model weights during training.
- **Deployment on GKE**: Deploying the trained model on **Google Kubernetes Engine (GKE)**.

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

## Deployment Steps

Follow the steps below to deploy the trained model on **GKE**:

### 1. Create ConfigMap
Configure necessary environment variables for the deployment:
```bash
kubectl create configmap ml-model-config \
  --from-literal=gcp_bucket_name=model_weights_demo \
  --from-literal=bigquery_dataset=model_prediction \
  --from-literal=TRUE
2. Create Secret
Provide the GCP service account key as a Kubernetes secret:

bash
Copy code
kubectl create secret generic gcp-service-account-key \
  --from-file=key.json=/Users/rankum/Downloads/perfect-science-437706-m0-dcc2dbaf6c93.json
3. Apply Kubernetes Deployment
Deploy the model using the provided Kubernetes deployment YAML file:

bash
Copy code
kubectl apply -f k8s/deployement.yaml
Infrastructure Management with Terraform
The infrastructure for this project is managed using Terraform. Follow these steps to set up and manage the infrastructure:

Initialize Terraform
bash
Copy code
terraform init
Plan the Infrastructure
bash
Copy code
terraform plan -var-file="prod-variables-file.tfvars" "-lock=false"
Apply Changes
bash
Copy code
terraform apply -var-file="prod-variables-file.tfvars" "-lock=false"
Destroy Resources
If needed, destroy the resources:

bash
Copy code
terraform destroy -var-file="prod-variables-file.tfvars" "-lock=false"
Project Highlights
Scalable Deployment: The project leverages GKE for efficient, scalable deployment of the machine learning model.
Infrastructure as Code: With Terraform, infrastructure management is simplified and reproducible.
Secure Configurations: Sensitive data is handled securely using Kubernetes ConfigMaps and Secrets.
Next Steps
Monitoring: Integrate tools like Prometheus and Grafana for monitoring the model deployment.
CI/CD: Implement CI/CD pipelines for automating the deployment process.
Model Updates: Set up pipelines for automatically deploying updated models.
Acknowledgments
This project was developed using Google Cloud Platform services and open-source tools like Kubernetes and Terraform.

vbnet
Copy code




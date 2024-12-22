# Scalable Model Deployment

This project focuses on training and deploying a scalable machine learning model using TensorFlow and Keras. The project includes data preprocessing, model training, and model checkpointing, and model deployment on gke.




# Prediction Table Schema

| **Field Name**           | **Type**     | **Mode**    | **Key** | **Collation** | **Default Value** | **Policy Tags** | **Description** |
|---------------------------|--------------|-------------|---------|---------------|-------------------|-----------------|-----------------|
| `prediction_id`           | STRING       | REQUIRED    | -       | -             | -                 | -               | -               |
| `model_version`           | STRING       | REQUIRED    | -       | -             | -                 | -               | -               |
| `input_shape`             | STRING       | REQUIRED    | -       | -             | -                 | -               | -               |
| `predicted_class`         | INTEGER      | REQUIRED    | -       | -             | -                 | -               | -               |
| `class_probabilities`     | FLOAT        | REPEATED    | -       | -             | -                 | -               | -               |
| `created_at`              | TIMESTAMP    | REQUIRED    | -       | -             | -                 | -               | -               |
| `processing_time_ms`      | FLOAT        | REQUIRED    | -       | -             | -                 | -               | -               |


kubectl create configmap ml-model-config \
    --from-literal=gcp_bucket_name=model_weights_demo \
    --from-literal=bigquery_dataset=model_prediction \
—from-literal=TRUE

kubectl create secret generic gcp-service-account-key \
  --from-file=key.json=/Users/rankum/Downloads/perfect-science-437706-m0-dcc2dbaf6c93.json

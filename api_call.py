import requests
import numpy as np

# Create sample image
image = np.random.rand(28, 28)

# Prepare request
url = "http://34.31.11.123/predict"
payload = {
    "image": image.tolist(),
    "model_version": "v1"
}

# Make request
response = requests.post(url, json=payload)
result = response.json()
print(result)
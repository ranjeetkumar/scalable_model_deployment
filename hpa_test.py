import requests
import numpy as np
import concurrent.futures
import time

# API URL
url = "http://34.31.11.123/predict"

# Create sample image data
def generate_sample_image():
    return np.random.rand(28, 28)

# Function to make a prediction request
def make_prediction_request():
    image = generate_sample_image()  # Generate new image data
    payload = {
        "image": image.tolist(),
        "model_version": "v1"
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Error during API call: {e}")

# Function to send multiple requests in parallel
def load_test(num_requests=100):
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Send requests in parallel
        futures = [executor.submit(make_prediction_request) for _ in range(num_requests)]
        
        # Wait for all the requests to complete
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    print(f"Load test completed: {num_requests} requests in {end_time - start_time:.2f} seconds.")

# Run the load test
if __name__ == "__main__":
    print("Starting load test...")
    load_test(num_requests=10000000)  # Adjust the number of requests to simulate load

# Prediction client
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# Key class for azure
from msrest.authentication import ApiKeyCredentials
# dotenv to load key
from dotenv import load_dotenv
# Import os to read environment variables
import os

# Load the key and endpoint values
load_dotenv()
# Set the values into variables
key = os.getenv('a7dcb2f74404411c81e1be9b6ee31b57')
endpoint = os.getenv('https://matalanga.cognitiveservices.azure.com/')
project_id = os.getenv('11f8cd33-5b49-4e7f-afb8-7f1102505af7')
published_name = os.getenv('PUBLISHED_ITERATION_NAME')


# Setup credentials for client
credentials = ApiKeyCredentials(in_headers={'Prediction-key':key})

# Create client, which will be used to make predictions
client = CustomVisionPredictionClient(endpoint, credentials)

# Open the test file
with open('testing-images/american-staffordshire-terrier-10.jpg', 'rb') as image:
    # Perform the prediction
    results = client.classify_image(project_id, published_name, image.read())

    # Because there could be multiple predictions, we loop through each one
    for prediction in results.predictions:
        # Display the name of the breed, and the probability percentage
        print(f'{prediction.tag_name}: {(prediction.probability):.2%}')
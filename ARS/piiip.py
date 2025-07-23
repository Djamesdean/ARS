import openwakeword.utils
import os

# Define the path to your models directory
models_dir = os.path.join(os.getcwd(), 'models')

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# Download the "alexa" pre-trained model to the specified directory

openwakeword.utils.download_models(model_names=["alexa"])
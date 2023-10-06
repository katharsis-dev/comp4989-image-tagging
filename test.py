import tensorflow as tf

# List available models for image classification
available_models = tf.keras.applications.available_models

# Print the list of available models
for model_name in available_models:
    print(model_name)


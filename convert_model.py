from tensorflow.keras.models import load_model

# Load the existing .h5 model
model = load_model("updated_model.h5", compile=False)


# Convert the model to JSON format
model_json = model.to_json()

# Save the JSON model structure
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights separately
model.save_weights("model_weights.weights.h5")


print("✅ Model successfully converted to JSON format!")

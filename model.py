from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Download the model
model_name = "A2H0H0R1/resnet-50-plant-disease"

# Load model
model = AutoModelForImageClassification.from_pretrained(model_name)

# Load feature extractor (needed for class labels)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
# Save model
model.save_pretrained("./plant_disease_model")

# Save feature extractor
feature_extractor.save_pretrained("./plant_disease_model")

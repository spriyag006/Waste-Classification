import os
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import json
 
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Fallback if only full tensorflow is installed locally
    from tensorflow.lite import Interpreter
 
app = Flask(__name__)
 
 
# Load TFLite model
interpreter = Interpreter(model_path="model/waste_model.tflite")
interpreter.allocate_tensors()
 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
 
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)
 
print("Model loaded successfully!")
print("Classes:", class_names)
 
 
@app.route("/")
def home():
    return render_template("index.html")
 
 
@app.route("/predict", methods=["POST"])
def predict():
 
    if "waste" not in request.files:
        return "No image uploaded"
 
    file = request.files["waste"]
 
    if file.filename == "":
        return "No image selected"
 
    image = Image.open(file).convert("RGB")
 
    image = image.resize((224, 224))
 
    image_array = np.array(image, dtype=np.float32)
 
    image_array = np.expand_dims(image_array, axis=0)
 
    # Prediction via TFLite interpreter
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
 
    predicted_index = np.argmax(predictions[0])
 
    predicted_class = class_names[predicted_index]
 
    confidence = (
        float(predictions[0][predicted_index]) * 100
    )
 
    print("\nPrediction probabilities:")
 
    for i, name in enumerate(class_names):
        print(
            f"{name}: "
            f"{predictions[0][i] * 100:.2f}%"
        )
 
    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
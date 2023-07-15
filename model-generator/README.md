# Generate ONNX from tflite file:
python -m tf2onnx.convert --tflite q_aware_model.tflite --output q_aware_model.onnx

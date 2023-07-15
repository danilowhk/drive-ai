import onnx
import onnx.numpy_helper
from keras.datasets import mnist
from scipy.ndimage import zoom
import os
import numpy as np

def resize_images(images):
    return np.array([zoom(image, 0.5) for image in images])

(_, _), (x_test_image, y_test_label) = mnist.load_data()
x_test_image = resize_images(x_test_image)
x_test_image_norm = (x_test_image / 255.0 * 255 - 128).astype(np.int8)

# Load the ONNX model
onnx_model = onnx.load("q_aware_model.onnx")

# Extract the weights and biases from the ONNX model
# Extract the weights and biases from the ONNX model
weights = {}
for initializer in onnx_model.graph.initializer:
    weights[initializer.name] = onnx.numpy_helper.to_array(initializer)

# Fetch weights and biases
weights_and_biases = {key: value for key, value in weights.items() if 'MatMul' in key or 'BiasAdd' in key}

# Split weights and biases based on layer
fc1_weights = [value for key, value in weights_and_biases.items() if 'quant_dense/MatMul' in key][0]
fc1_bias = [value for key, value in weights_and_biases.items() if 'quant_dense/BiasAdd' in key][0]
fc2_weights = [value for key, value in weights_and_biases.items() if 'quant_dense_1/MatMul' in key][0]
fc2_bias = [value for key, value in weights_and_biases.items() if 'quant_dense_1/BiasAdd' in key][0]

tensors = {
    "input": x_test_image[0].flatten(),
    "fc1_weights": fc1_weights, 
    "fc1_bias": fc1_bias, 
    "fc2_weights": fc2_weights, 
    "fc2_bias": fc2_bias
}

# Create the directory if it doesn't exist
os.makedirs('cairo-files/weights-biases', exist_ok=True)

for tensor_name, tensor in tensors.items():
    with open(os.path.join('cairo-files', 'weights-biases', f"{tensor_name}.cairo"), "w") as f:
        f.write(
            "use array::ArrayTrait;\n" +
            "use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};\n" +
            "use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;\n" +
            "use orion::numbers::fixed_point::core::FixedImpl;\n" +
            "use orion::numbers::signed_integer::i32::i32;\n\n" +
            "fn {0}() -> Tensor<i32> ".format(tensor_name) + "{\n" +
            "    let mut shape = ArrayTrait::<usize>::new();\n"
        )
        for dim in tensor.shape:
            f.write("    shape.append({0});\n".format(dim))
        f.write(
            "    let mut data = ArrayTrait::<i32>::new();\n"
        )
        for val in np.nditer(tensor.flatten()):
            f.write("    data.append(i32 {{ mag: {0}, sign: {1} }});\n".format(abs(int(val)), str(val < 0).lower()))
        f.write(
            "let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; \n" +
            "    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))\n" +
            "}\n"
        )
      
with open(os.path.join('cairo-files', 'generated.cairo'), 'w') as f:
    for param_name in tensors.keys():
        f.write(f"mod {param_name};\n")

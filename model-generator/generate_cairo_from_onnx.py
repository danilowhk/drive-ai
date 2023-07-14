import onnx
import numpy as np
import os

# ONNX model file
model_file = 'model_cnn.onnx'

# Extract base model name without extension
model_name = os.path.splitext(os.path.basename(model_file))[0]

# Load ONNX model
model = onnx.load(model_file)
print(model)

# Generate Cairo files 
def process_tensor(initializer):
    if 'weight' in initializer.name:
        return np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
    else:
        return np.frombuffer(initializer.raw_data, dtype=np.float32)

tensors = {}
for initializer in model.graph.initializer:
    name = initializer.name.replace('.', '_')
    tensors[name] = process_tensor(initializer)  

print(tensors)

# Use model name in directory paths
cairo_files_dir = f'files/{model_name}_generated_from_onnx'
os.makedirs(cairo_files_dir, exist_ok=True)

for tensor_name, tensor in tensors.items():
    with open(os.path.join(cairo_files_dir, f"{tensor_name}.cairo"), "w") as f:
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

with open(os.path.join('files', f'{model_name}_generated_from_onnx.cairo'), 'w') as f:
    for param_name in tensors.keys():
        f.write(f"mod {param_name};\n")

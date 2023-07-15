use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn sequential_quant_dense_1_BiasAdd_ReadVariableOp() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(40);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 253, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 36, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 16, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 186, sign: false });
    data.append(i32 { mag: 253, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 79, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 253, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 253, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 182, sign: false });
    data.append(i32 { mag: 250, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 255, sign: false });
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
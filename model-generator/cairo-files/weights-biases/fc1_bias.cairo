use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 40, sign: true });
    data.append(i32 { mag: 3088, sign: false });
    data.append(i32 { mag: 5483, sign: false });
    data.append(i32 { mag: 10301, sign: true });
    data.append(i32 { mag: 4480, sign: false });
    data.append(i32 { mag: 2592, sign: false });
    data.append(i32 { mag: 7942, sign: false });
    data.append(i32 { mag: 2091, sign: false });
    data.append(i32 { mag: 5340, sign: false });
    data.append(i32 { mag: 1584, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}

"""CUDA implementations of the Module 4 convolution operators."""

from typing import Tuple

from numba import cuda

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function


THREADS_PER_BLOCK = 128


@cuda.jit
def _tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input_storage,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """Write one 1D convolution output element per CUDA thread."""
    ordinal = cuda.grid(1)
    if ordinal >= out_size:
        return

    out_channels = out_shape[1]
    out_width = out_shape[2]
    in_channels = input_shape[1]
    width = input_shape[2]
    kernel_width = weight_shape[2]

    batch_index = ordinal // (out_channels * out_width)
    position = ordinal % (out_channels * out_width)
    out_channel = position // out_width
    out_col = position % out_width

    accumulator = 0.0
    for in_channel in range(in_channels):
        for kernel_col in range(kernel_width):
            if reverse:
                input_col = out_col - kernel_col
            else:
                input_col = out_col + kernel_col

            if 0 <= input_col < width:
                input_position = (
                    batch_index * input_strides[0]
                    + in_channel * input_strides[1]
                    + input_col * input_strides[2]
                )
                weight_position = (
                    out_channel * weight_strides[0]
                    + in_channel * weight_strides[1]
                    + kernel_col * weight_strides[2]
                )
                accumulator += (
                    input_storage[input_position] * weight[weight_position]
                )

    output_position = (
        batch_index * out_strides[0]
        + out_channel * out_strides[1]
        + out_col * out_strides[2]
    )
    out[output_position] = accumulator


def tensor_conv1d(*args) -> None:
    """Launch the CUDA 1D convolution kernel."""
    out_size = args[3]
    blocks = (out_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _tensor_conv1d[blocks, THREADS_PER_BLOCK](*args)


class CudaConv1dFun(Function):
    """Autodifferentiable CUDA 1D convolution."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        assert input.backend.cuda and weight.backend.cuda
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, weight_in_channels, _ = weight.shape
        assert in_channels == weight_in_channels

        output = input.zeros((batch, out_channels, width))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, _, kernel_width = weight.shape

        grad_weight = grad_output.zeros(
            (in_channels, out_channels, kernel_width)
        )
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *input.permute(1, 0, 2).tuple(),
            *grad_output.permute(1, 0, 2).tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, width))
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *weight.permute(1, 0, 2).tuple(),
            True,
        )
        return grad_input, grad_weight


@cuda.jit
def _tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input_storage,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """Write one 2D convolution output element per CUDA thread."""
    ordinal = cuda.grid(1)
    if ordinal >= out_size:
        return

    out_channels = out_shape[1]
    out_height = out_shape[2]
    out_width = out_shape[3]
    in_channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    kernel_height = weight_shape[2]
    kernel_width = weight_shape[3]

    batch_index = ordinal // (out_channels * out_height * out_width)
    position = ordinal % (out_channels * out_height * out_width)
    out_channel = position // (out_height * out_width)
    spatial_position = position % (out_height * out_width)
    out_row = spatial_position // out_width
    out_col = spatial_position % out_width

    accumulator = 0.0
    for in_channel in range(in_channels):
        for kernel_row in range(kernel_height):
            if reverse:
                input_row = out_row - kernel_row
            else:
                input_row = out_row + kernel_row

            if 0 <= input_row < height:
                for kernel_col in range(kernel_width):
                    if reverse:
                        input_col = out_col - kernel_col
                    else:
                        input_col = out_col + kernel_col

                    if 0 <= input_col < width:
                        input_position = (
                            batch_index * input_strides[0]
                            + in_channel * input_strides[1]
                            + input_row * input_strides[2]
                            + input_col * input_strides[3]
                        )
                        weight_position = (
                            out_channel * weight_strides[0]
                            + in_channel * weight_strides[1]
                            + kernel_row * weight_strides[2]
                            + kernel_col * weight_strides[3]
                        )
                        accumulator += (
                            input_storage[input_position]
                            * weight[weight_position]
                        )

    output_position = (
        batch_index * out_strides[0]
        + out_channel * out_strides[1]
        + out_row * out_strides[2]
        + out_col * out_strides[3]
    )
    out[output_position] = accumulator


def tensor_conv2d(*args) -> None:
    """Launch the CUDA 2D convolution kernel."""
    out_size = args[3]
    blocks = (out_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _tensor_conv2d[blocks, THREADS_PER_BLOCK](*args)


class CudaConv2dFun(Function):
    """Autodifferentiable CUDA 2D convolution."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        assert input.backend.cuda and weight.backend.cuda
        ctx.save_for_backward(input, weight)
        batch, in_channels, height, width = input.shape
        out_channels, weight_in_channels, _, _ = weight.shape
        assert in_channels == weight_in_channels

        output = input.zeros((batch, out_channels, height, width))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, height, width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        grad_weight = grad_output.zeros(
            (in_channels, out_channels, kernel_height, kernel_width)
        )
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *input.permute(1, 0, 2, 3).tuple(),
            *grad_output.permute(1, 0, 2, 3).tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, height, width))
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *weight.permute(1, 0, 2, 3).tuple(),
            True,
        )
        return grad_input, grad_weight


cuda_conv1d = CudaConv1dFun.apply
cuda_conv2d = CudaConv2dFun.apply

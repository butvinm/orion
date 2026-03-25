from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orion_compiler.nn.linear import Conv2d, Linear, LinearTransform
    from orion_compiler.nn.normalization import BatchNorm1d, BatchNorm2d

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -------------------#
#   Packing Logic   #
# -------------------#


def pack_conv2d(
    conv_layer: Conv2d, last: bool
) -> tuple[dict[tuple[int, int], dict[int, list[float]]], int]:
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    weight = conv_layer.on_weight
    if conv_layer.groups > 1:
        weight = resolve_grouped_conv(conv_layer)

    toeplitz = construct_conv2d_toeplitz(conv_layer, weight)
    diagonals, output_rotations = diagonalize(toeplitz, slots, embed_method, last)

    return diagonals, output_rotations


def construct_conv2d_toeplitz(conv_layer: Conv2d, weight: torch.Tensor) -> sp.csr_matrix:
    assert conv_layer.fhe_input_shape is not None
    assert conv_layer.fhe_output_shape is not None
    assert conv_layer.output_shape is not None
    N, on_Ci, on_Hi, on_Wi = conv_layer.fhe_input_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    Ho, Wo = conv_layer.output_shape[2:]

    P = conv_layer.padding[0]
    D = conv_layer.dilation[0]
    iG = conv_layer.input_gap
    oG = conv_layer.output_gap
    kW, kH = weight.shape[2:]

    def compute_first_kernel_position() -> torch.Tensor:
        mpx_anchors = valid_image_indices[:, :iG, :iG].reshape(-1, 1)

        row_idxs = torch.arange(0, kH * D * iG, D * iG).reshape(-1, 1)
        col_idxs = torch.arange(0, kW * D * iG, D * iG)
        kernel_offsets = valid_image_indices[0, row_idxs, col_idxs].flatten()

        img_pixels_touched = mpx_anchors + kernel_offsets
        return img_pixels_touched.flatten()

    def compute_row_interchange_map() -> torch.Tensor:
        output_indices = torch.arange(on_Ho * on_Wo).reshape(on_Ho, on_Wo)

        start_indices = output_indices[:oG, :oG].flatten()
        corner_indices = output_indices[0 : (Ho * oG) : oG, 0 : (Wo * oG) : oG].reshape(-1, 1)
        return corner_indices + start_indices

    # Padded input dimensions with multiplexing
    Hi_pad = on_Hi + 2 * P * iG
    Wi_pad = on_Wi + 2 * P * iG

    # Initialize our sparse Toeplitz matrix
    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * Hi_pad * Wi_pad
    toeplitz = sp.lil_matrix((n_rows, n_cols), dtype="f")

    # Create an index grid for the padded input image.
    valid_image_indices = torch.arange(n_cols).reshape(on_Ci, Hi_pad, Wi_pad)

    # Pad the kernel's input and output channels to the nearest multiple
    # of gap^2 to ensure that multiplexing works.
    kernel = torch.zeros(on_Co * oG**2, on_Ci * iG**2, kW, kH)
    kernel[: weight.shape[0], : weight.shape[1], ...] = weight

    # All the indices the kernel initially touches
    initial_kernel_position = compute_first_kernel_position()

    # Create our row-interchange map that dictates how we permute rows for
    # optimal packing. Also return all indices that the first top-left filter
    # value touches throughout the convolution.
    row_map = compute_row_interchange_map()
    corner_indices = valid_image_indices[0, 0 : (Ho * oG) : oG, 0 : (Wo * oG) : oG].flatten()

    # Create offsets for the multiplexed output channels.
    out_channels = (torch.arange(on_Co) * (on_Ho * on_Wo)).reshape(on_Co, 1)

    # Flattened kernel populates rows of our Toeplitz matrix
    kernel_flat = kernel.reshape(kernel.shape[0], -1)

    # Iterate over all positions that the top-left kernel element can touch
    # populating the correct (permuted) rows of our Toeplitz matrix.
    for i, start_idx in enumerate(corner_indices):
        rows = (row_map[i] + out_channels).reshape(-1, 1)
        cols = initial_kernel_position + start_idx
        toeplitz[rows, cols] = kernel_flat

    # Keep only the columns corresponding to the non-padded input image.
    row_idxs = torch.arange(P * iG, P * iG + on_Hi).reshape(-1, 1)
    col_idxs = torch.arange(P * iG, P * iG + on_Wi)
    image_indices = valid_image_indices[:, row_idxs, col_idxs].flatten()
    toeplitz = toeplitz.tocsc()[:, image_indices]

    # Support batching
    toeplitz = sp.kron(sp.eye(N, dtype="f"), toeplitz, format="csr")
    return toeplitz


def construct_conv2d_bias(conv_layer: Conv2d) -> torch.Tensor:
    assert conv_layer.output_shape is not None
    assert conv_layer.fhe_output_shape is not None
    N, Co, Ho, Wo = conv_layer.output_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]

    bias = conv_layer.on_bias
    bias = bias.repeat_interleave(Ho * Wo)
    bias = bias.reshape(1, Co, Ho, Wo)
    bias_multiplexed = multiplex(bias, conv_layer.output_gap).squeeze(0)

    mC, mH, mW = bias_multiplexed.shape
    bias_vector = torch.zeros(on_Co, on_Ho, on_Wo)
    bias_vector[:mC, :mH, :mW] = bias_multiplexed
    bias_vector = bias_vector.flatten().repeat(N)

    return bias_vector


def pack_linear(
    linear_layer: Linear, last: bool
) -> tuple[dict[tuple[int, int], dict[int, list[float]]], int]:
    slots = linear_layer.scheme.params.get_slots()
    embed_method = linear_layer.scheme.params.get_embedding_method()

    weight = construct_linear_matrix(linear_layer)
    diagonals, output_rotations = diagonalize(weight, slots, embed_method, last)
    return diagonals, output_rotations


def construct_linear_matrix(linear_layer: Linear) -> sp.csr_matrix:
    assert linear_layer.input_shape is not None
    if len(linear_layer.input_shape) == 2:
        N = linear_layer.input_shape[0]
        matrix = linear_layer.on_weight
    else:  # Prior layer was not a linear layer
        out_features = linear_layer.out_features
        input_gap = linear_layer.input_gap
        N, Ci, Hi, Wi = linear_layer.input_shape
        assert linear_layer.fhe_input_shape is not None
        on_Ci, on_Hi, on_Wi = linear_layer.fhe_input_shape[1:]

        reshaped = linear_layer.on_weight.reshape(out_features, Ci, Hi, Wi)
        reshaped = multiplex(reshaped, input_gap)

        matrix = torch.zeros(out_features, on_Ci, on_Hi, on_Wi)
        matrix[..., : Hi * input_gap, : Wi * input_gap] = reshaped
        matrix = matrix.reshape(out_features, -1)

    matrix = torch.kron(torch.eye(N), matrix)
    matrix_sparse = sp.csr_matrix(matrix.cpu().numpy())
    return matrix_sparse


def construct_linear_bias(linear_layer: LinearTransform) -> torch.Tensor:
    assert linear_layer.input_shape is not None
    N = linear_layer.input_shape[0]
    result: torch.Tensor = linear_layer.on_bias.repeat(N)
    return result


# -----------------------------#
#       Helper Functions      #
# -----------------------------#


def multiplex(matrix: torch.Tensor, gap: int) -> torch.Tensor:
    N, Ci, Hi, Wi = matrix.shape
    Co = math.ceil(Ci / (gap**2))

    # Pad the tensor to have channels divisible by gap^2
    padded = torch.zeros(N, Co * gap**2, Hi, Wi)
    padded[:, :Ci, ...] = matrix
    result: torch.Tensor = F.pixel_shuffle(padded, gap)  # multiplexed
    return result


def resolve_grouped_conv(conv_layer: Conv2d) -> torch.Tensor:
    on_weight = conv_layer.on_weight.repeat(1, conv_layer.groups, 1, 1)

    # Zero out input channels to support arbitrary groups
    mask = torch.zeros_like(on_weight)
    Ci_per_group = conv_layer.in_channels // conv_layer.groups
    Co_per_group = conv_layer.out_channels // conv_layer.groups

    for i in range(conv_layer.groups):
        mask[
            i * Co_per_group : (i + 1) * Co_per_group,
            i * Ci_per_group : (i + 1) * Ci_per_group,
            ...,
        ] = 1

    result: torch.Tensor = on_weight * mask
    return result


def _extract_diagonals_sparse(
    block_sparse: sp.spmatrix,
    block_height: int,
    num_slots: int,
    reps: int,
) -> dict[int, list[float]]:
    """Extract cyclic diagonals directly from sparse block. O(nnz) memory.

    For a block of shape (block_height, num_slots), cyclic diagonal ``d`` at
    position ``j`` is defined as ``block[j % block_height, (d + j) % num_slots]``.
    Each non-zero at ``(r, c)`` maps to exactly one diagonal:
    ``d = (c - r) % block_height``, at position
    ``j = r + ((c - r) // block_height % reps) * block_height``.
    """
    coo = block_sparse.tocoo()
    if coo.nnz == 0:
        return {}

    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    data = coo.data.astype(np.float64)

    # Each non-zero (r, c, v) maps to exactly one (diag_index, position).
    diff = cols - rows
    diag_indices = diff % block_height
    if reps == 1:
        positions = rows
    else:
        k = (diff // block_height) % reps
        positions = rows + k * block_height

    # Sort by diagonal index so we can split into groups efficiently
    order = np.argsort(diag_indices, kind="mergesort")
    sorted_diags = diag_indices[order]
    sorted_pos = positions[order]
    sorted_vals = data[order]

    # Find unique diagonals and split points
    unique_diags, first_idx = np.unique(sorted_diags, return_index=True)
    split_points = first_idx[1:]

    result: dict[int, list[float]] = {}
    groups = np.split(np.arange(len(order)), split_points)
    for d, group in zip(unique_diags, groups, strict=False):
        diag_vec = np.zeros(num_slots, dtype=np.float64)
        diag_vec[sorted_pos[group]] = sorted_vals[group]
        if np.any(diag_vec != 0):
            result[int(d)] = diag_vec.tolist()

    return result


def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
) -> tuple[dict[tuple[int, int], dict[int, list[float]]], int]:
    """
    For each (slots, slots) block of the input matrix, this function
    extracts the generalized diagonals and stores them in a dictionary.
    Each key ((i,j)) in the dictionary block_{i,j}, and the value is
    another dictionary mapping diagonal indices to their values.

    Args:
        matrix (scipy.sparse.csr_matrix): A 4D tensor representing a weight matrix
            for a fully-connected or convolutional layer. The shape must
            conform to (num_blocks_y, num_blocks_x, slots, slots).
        num_slots (int): The number of SIMD plaintext slots, dictating the
            block size.
        embed_method (str): The embedding method to use.
        is_last_layer (bool): Whether this is the last layer.

    Returns:
        A tuple of (diagonals_by_block, output_rotations) where
        diagonals_by_block is a dictionary mapping block coordinates to
        diagonal dictionaries, and output_rotations is the number of
        output rotations needed.

    Examples:
        >>> matrix = torch.tensor([[[[ 0,  1,  2,  3],
                                     [ 4,  5,  6,  7],
                                     [ 8,  9, 10, 11],
                                     [12, 13, 14, 15]]]])
        >>> # Example with slots=4, showing processing of a single block
        >>> print(diagonalize(matrix, slots=4))
        {(0, 0): {0: [0., 5., 10., 15.],
                  1: [1., 6., 11., 12.],
                  2: [2., 7., 8., 13.],
                  3: [3., 4., 9., 14.]}}

        >>> # Example with slots=2, showing processing of four blocks or
              sub-matrices
        >>> print(diagonalize(matrix, slots=2))
        {(0, 0): {0: [0., 5.],
                  1: [1., 4.]},
         (0, 1): {0: [2., 7.],
                  1: [3., 6.]},
         (1, 0): {0: [8., 13.],
                  1: [9., 12.]},
         (1, 1): {0: [10., 15.],
                  1: [11., 14.]}}
    """

    matrix_height, matrix_width = matrix.shape
    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)
    logger.debug("embed method: %s", embed_method)
    logger.debug("original matrix shape: %s", matrix.shape)
    logger.debug("# blocks (rows, cols) = %s", (num_block_rows, num_block_cols))

    if num_block_rows == 1 and embed_method == "hybrid" and not is_last_layer:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    # Inflate dimensions of the sparse matrix
    matrix.resize(num_block_rows * block_height, num_block_cols * num_slots)

    logger.debug("resized matrix shape: %s", matrix.shape)
    logger.debug("# output rotations: %d", output_rotations)

    reps = max(1, num_slots // block_height)

    diagonals_by_block: dict[tuple[int, int], dict[int, list[float]]] = {}
    total_diagonals = 0

    # Process each block
    progress_bar = tqdm(
        total=num_block_rows * num_block_cols,
        desc="|    Processing blocks",
        leave=False,
    )
    start_time = time.time()
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            row_start = num_slots * block_row
            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start : row_start + block_height,
                col_start : col_start + num_slots,
            ]

            nonzero_diagonals = _extract_diagonals_sparse(
                block_sparse,
                block_height,
                num_slots,
                reps,
            )

            total_diagonals += len(nonzero_diagonals)
            diagonals_by_block[(block_row, block_col)] = nonzero_diagonals or {
                0: [0.0] * num_slots
            }

            progress_bar.set_postfix(
                {
                    "Current Block": f"({block_row},{block_col})",
                    "Total Diagonals": total_diagonals,
                }
            )
            progress_bar.update(1)

    progress_bar.close()
    elapsed_time = time.time() - start_time
    logger.debug("time to pack (s): %.2f", elapsed_time)
    logger.debug("# diagonals = %d", total_diagonals)

    return diagonals_by_block, output_rotations


def plot_toeplitz(matrix: sp.csr_matrix | Any, save_path: str = "") -> None:
    if isinstance(matrix, sp.csr_matrix):
        matrix = matrix.todense()

    if matrix.ndim != 2:
        raise ValueError(f"Cannot plot matrix of dimension {matrix.ndim}")

    plt.imshow(matrix)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# ---------------------#
#   BatchNorm Logic   #
# ---------------------#


def pack_bn1d(
    bn1d_layer: BatchNorm1d,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    assert bn1d_layer.input_shape is not None
    N = bn1d_layer.input_shape[0]
    on_running_mean = bn1d_layer.on_running_mean
    on_inv_running_std = 1 / torch.sqrt(bn1d_layer.on_running_var + bn1d_layer.eps)
    on_weight = bn1d_layer.on_weight if bn1d_layer.affine else None
    on_bias = bn1d_layer.on_bias if bn1d_layer.affine else None

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N),
        on_weight.flatten().repeat(N) if on_weight is not None else None,
        on_bias.flatten().repeat(N) if on_bias is not None else None,
    )


def pack_bn2d(
    bn2d_layer: BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    assert bn2d_layer.input_shape is not None
    assert bn2d_layer.fhe_input_shape is not None
    N, Ci, Hi, Wi = bn2d_layer.input_shape
    on_Ci, on_Hi, on_Wi = bn2d_layer.fhe_input_shape[1:]

    on_running_mean = torch.zeros(on_Ci, on_Hi, on_Wi)
    on_inv_running_std = torch.zeros(on_Ci, on_Hi, on_Wi)

    mean = bn2d_layer.on_running_mean.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
    var = bn2d_layer.on_running_var.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

    mean_mpx = multiplex(mean, bn2d_layer.input_gap).squeeze(0)
    var_mpx = multiplex(var, bn2d_layer.input_gap).squeeze(0)

    mC, mH, mW = mean_mpx.shape
    on_running_mean[:mC, :mH, :mW] = mean_mpx
    on_inv_running_std[:mC, :mH, :mW] = 1 / torch.sqrt(var_mpx + bn2d_layer.eps)

    on_weight = None
    on_bias = None
    if bn2d_layer.affine:
        on_weight = torch.zeros(on_Ci, on_Hi, on_Wi)
        on_bias = torch.zeros(on_Ci, on_Hi, on_Wi)

        weight = bn2d_layer.on_weight.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
        bias = bn2d_layer.on_bias.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

        weight_mpx = multiplex(weight, bn2d_layer.input_gap).squeeze(0)
        bias_mpx = multiplex(bias, bn2d_layer.input_gap).squeeze(0)

        mC, mH, mW = weight_mpx.shape
        on_weight[:mC, :mH, :mW] = weight_mpx
        on_bias[:mC, :mH, :mW] = bias_mpx

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N),
        on_weight.flatten().repeat(N) if on_weight is not None else None,
        on_bias.flatten().repeat(N) if on_bias is not None else None,
    )

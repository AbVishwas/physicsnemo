# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint

from physicsnemo.models.diffusion import (
    AttentionOp,
    FourierEmbedding,
    Linear,
    PositionalEmbedding,
    weight_init,
)
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


class Conv3d(torch.nn.Module):
    """
    3D convolution layer with optional upsampling and downsampling.

    This layer implements a 3D convolution operation with optional bilinear
    resampling (upsampling or downsampling) capabilities. It supports both
    fused and non-fused resampling modes for efficiency.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel : int
        Kernel size for the convolution (applied uniformly across all spatial dimensions).
    bias : bool, optional
        Whether to include a learnable bias term, by default True.
    up : bool, optional
        Whether to apply 2x upsampling before/after convolution, by default False.
    down : bool, optional
        Whether to apply 2x downsampling before/after convolution, by default False.
    resample_filter : List[int], optional
        1D filter coefficients for bilinear resampling, by default [1, 1].
        The 3D filter is constructed as outer product of this 1D filter.
    fused_resample : bool, optional
        Whether to fuse resampling with convolution for efficiency, by default False.
    init_mode : str, optional
        Weight initialization mode, by default "kaiming_normal".
    init_weight : float, optional
        Multiplier for weight initialization, by default 1.0.
    init_bias : float, optional
        Multiplier for bias initialization, by default 0.0.

    Raises
    ------
    ValueError
        If both `up` and `down` are set to True simultaneously.

    Note
    ----
    When `fused_resample=True`, the resampling operation is combined with
    convolution for improved computational efficiency. The resample filter
    is constructed as a 3D separable filter from the 1D coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel * kernel,
            fan_out=out_channels * kernel * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init(
                    [out_channels, in_channels, kernel, kernel, kernel], **init_kwargs
                )
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = (f.ger(f).unsqueeze(2) * f.view(1, 1, -1)).unsqueeze(0).unsqueeze(
            1
        ) / f.sum().pow(3)  # for 3D, should be ^3
        self.register_buffer("resample_filter", f.contiguous() if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose3d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv3d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv3d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv3d(
                x,
                f.tile([self.out_channels, 1, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose3d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv3d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv3d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.

    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if self.training:
            # Use default torch implementation of GroupNorm for training
            # This does not support channels last memory format
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=self.weight.to(x.dtype),
                bias=self.bias.to(x.dtype),
                eps=self.eps,
            )
        else:
            # Use custom GroupNorm implementation that supports channels last
            # memory layout for inference
            dtype = x.dtype
            x = x.float()
            x = rearrange(x, "b (g c) d h w -> b g c d h w", g=self.num_groups)

            mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)  # added 5th dim
            var = x.var(dim=[2, 3, 4, 5], keepdim=True)

            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c d h w -> b (g c) d h w")

            weight = rearrange(self.weight, "c -> 1 c 1 1 1")
            bias = rearrange(self.bias, "c -> 1 c 1 1 1")
            x = x * weight + bias

            x = x.type(dtype)
        return x


class UNetBlock3D(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional
        If True, applies upsampling in the forward pass. By default False.
    down : bool, optional
        If True, applies downsampling in the forward pass. By default False.
    attention : bool, optional
        If True, enables the self-attention mechanism in the block. By default False.
    num_heads : int, optional
        Number of attention heads. If None, defaults to `out_channels // 64`.
    channels_per_head : int, optional
        Number of channels per attention head. By default 64.
    dropout : float, optional
        Dropout probability. By default 0.0.
    skip_scale : float, optional
        Scale factor applied to skip connections. By default 1.0.
    eps : float, optional
        Epsilon value used for normalization layers. By default 1e-5.
    resample_filter : List[int], optional
        Filter for resampling layers. By default [1, 1].
    resample_proj : bool, optional
        If True, resampling projection is enabled. By default False.
    adaptive_scale : bool, optional
        If True, uses adaptive scaling in the forward pass. By default True.
    init : dict, optional
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional
        Initialization parameters with zero weights for certain layers. By default
        {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv3d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        # torch.cuda.nvtx.range_push("UNetBlock3D")
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        # torch.cuda.nvtx.range_pop()
        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "SongUNet3D"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class SongUNet3D(Module):
    """
    3D U-Net diffusion backbone for volumetric data generation.

    This architecture extends the DDPM++ and NCSN++ models to 3D volumetric data,
    implementing a U-Net variant with optional self-attention, embeddings, and
    encoder-decoder components for generating 3D volumes.

    The model supports both conditional and unconditional generation with flexible
    architectural choices for encoder/decoder types, embedding types, and attention
    mechanisms. It can be configured for various 3D diffusion tasks including medical
    imaging, scientific simulations, and volumetric content generation.

    Architecture Overview
    ---------------------
    The model processes 3D volumetric inputs through:

    1. **Embedding Generation**: Maps noise levels, class labels, and augmentation
       labels to embeddings that condition the generation process.

    2. **U-Net Encoder**: A hierarchical encoder with multiple levels, where each level:
       - Downsamples spatial resolution by 2x (D, H, W dimensions)
       - Applies ``num_blocks`` residual blocks with conditioning
       - Optionally applies 3D self-attention at specified resolutions
       - Caches features for skip connections

    3. **U-Net Decoder**: Mirror of the encoder that:
       - Upsamples spatial resolution by 2x at each level
       - Combines features via skip connections from encoder
       - Produces the final denoised 3D volume

    Conditioning Mechanism
    ----------------------
    - **Noise labels**: Condition on diffusion timestep/noise level
    - **Class labels**: Optional vector-valued class conditioning
    - **Augmentation labels**: Optional data augmentation conditioning
    - **Image conditioning**: Concatenate conditioning volumes to input channels

    Parameters
    ----------
    img_resolution : Union[List[int], int]
        Spatial resolution of the volumetric data. Can be a single int for uniform
        resolution (D=H=W) or a list [D, H, W] for non-uniform dimensions.
        Note: Model can process different resolutions at inference, except when
        ``additive_pos_embed=True``.
    in_channels : int
        Number of input channels. Includes both latent channels and any additional
        channels for image-based conditioning. For unconditional models, should
        equal ``out_channels``.
    out_channels : int
        Number of output channels. Should match the number of channels in the
        latent state being denoised.
    label_dim : int, optional
        Dimension of vector-valued class labels for conditional generation.
        Set to 0 for unconditional generation, by default 0.
    augment_dim : int, optional
        Dimension of vector-valued augmentation labels. Set to 0 for no
        augmentation conditioning, by default 0.
    model_channels : int, optional
        Base channel multiplier for the network. Determines the number of
        channels at the first level, by default 128.
    channel_mult : List[int], optional
        Channel multipliers at each U-Net level. Length determines the number
        of levels. At level i, channels = ``channel_mult[i] * model_channels``,
        by default [1, 2, 2, 2].
    channel_mult_emb : int, optional
        Multiplier for embedding vector channels. Embedding dimension is
        ``model_channels * channel_mult_emb``, by default 4.
    num_blocks : int, optional
        Number of residual blocks at each U-Net level, by default 4.
    attn_resolutions : List[int], optional
        Spatial resolutions at which to apply 3D self-attention. Attention is
        applied when the feature map resolution matches these values exactly,
        by default [16].
    dropout : float, optional
        Dropout probability for intermediate activations in U-Net blocks,
        by default 0.10.
    label_dropout : float, optional
        Dropout probability for class labels, typically used for classifier-free
        guidance during training, by default 0.0.
    embedding_type : str, optional
        Noise level embedding type. Options: 'positional' (DDPM++), 'fourier'
        (NCSN++), or 'zero' (no embedding), by default "positional".
    channel_mult_noise : int, optional
        Channel multiplier for noise level embeddings. Noise embedding dimension
        is ``model_channels * channel_mult_noise``, by default 1.
    encoder_type : str, optional
        Encoder architecture variant. Options: 'standard' (DDPM++), 'residual'
        (NCSN++), or 'skip' (skip connections), by default "standard".
    decoder_type : str, optional
        Decoder architecture variant. Options: 'standard' or 'skip' (skip
        connections), by default "standard".
    resample_filter : List[int], optional
        1D filter coefficients for resampling operations. Use [1, 1] for DDPM++
        or [1, 3, 3, 1] for NCSN++, by default [1, 1].
    checkpoint_level : int, optional
        Number of levels to use gradient checkpointing. Higher values trade
        memory for computation. 0 disables checkpointing, by default 0.
    additive_pos_embed : bool, optional
        If True, adds learnable positional embeddings encoding spatial position
        (separate from temporal diffusion embeddings). When enabled, input
        resolution must match ``img_resolution``, by default False.

    Raises
    ------
    ValueError
        If ``embedding_type`` is not one of ['fourier', 'positional', 'zero'].
    ValueError
        If ``encoder_type`` is not one of ['standard', 'skip', 'residual'].
    ValueError
        If ``decoder_type`` is not one of ['standard', 'skip'].

    Note
    ----
    This is a 3D extension of the SongUNet architecture. The primary differences
    from the 2D version are:
    - All convolutions and attention operations work on 3D volumes (B, C, D, H, W)
    - Resampling filters are constructed as 3D separable filters
    - Self-attention operates on flattened 3D spatial dimensions

    See Also
    --------
    SongUNet : 2D variant of this architecture for image generation.
    EDMPrecond3D : Preconditioning wrapper for 3D diffusion models.

    References
    ----------
    .. [1] Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion
           probabilistic models. ICML 2021.
    .. [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
           & Poole, B. (2021). Score-based generative modeling through stochastic
           differential equations. ICLR 2021.

    Examples
    --------
    >>> # Create unconditional 3D diffusion model for 64^3 volumes
    >>> model = SongUNet3D(
    ...     img_resolution=64,
    ...     in_channels=4,
    ...     out_channels=4,
    ...     model_channels=128,
    ...     channel_mult=[1, 2, 2, 2],
    ...     num_blocks=4,
    ... )
    >>>
    >>> # Forward pass with noise conditioning
    >>> x = torch.randn(2, 4, 64, 64, 64)  # Noisy volumes
    >>> noise_labels = torch.randn(2, 128)  # Noise level embeddings
    >>> denoised = model(x, noise_labels)
    >>> denoised.shape
    torch.Size([2, 4, 64, 64, 64])
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise

        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))

        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Handle image resolution (now 3D)
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_z = self.img_shape_y = self.img_shape_x = img_resolution
        elif len(img_resolution) == 2:
            self.img_shape_y, self.img_shape_x = img_resolution
            self.img_shape_z = img_resolution[0]  # Default to same as y
        else:
            self.img_shape_z, self.img_shape_y, self.img_shape_x = img_resolution[:3]

        # Set checkpoint threshold based on resolution
        max_dimension = max(self.img_shape_x, self.img_shape_y, self.img_shape_z)
        self.checkpoint_threshold = (max_dimension >> checkpoint_level) + 1

        # Optional additive learned position embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(
                    1,
                    model_channels,
                    self.img_shape_z,
                    self.img_shape_y,
                    self.img_shape_x,
                )
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(in_features=label_dim, out_features=noise_channels, **init)
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )

        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv3d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv3d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv3d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv3d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv3d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv3d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping
        if self.embedding_type != "zero":
            emb = self.map_noise(noise_labels)
            emb = (
                emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            )  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device)
                        >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = F.silu(self.map_layer0(emb))
            emb = F.silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels), device=x.device
            )

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                if isinstance(block, UNetBlock3D):
                    if x.shape[-1] > self.checkpoint_threshold:
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(F.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                if (x.shape[-1] > self.checkpoint_threshold and "_block" in name) or (
                    x.shape[-1] > (self.checkpoint_threshold / 2) and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)
        return aux


if __name__ == "__main__":
    # Example usage
    model = SongUNet3D(
        img_resolution=8,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        augment_dim=0,
        model_channels=16,
        channel_mult=[1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0.0,
        embedding_type="positional",
        channel_mult_noise=1,
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1],
        checkpoint_level=0,
        additive_pos_embed=False,
    )
    print("Model created successfully.")

    x = torch.randn(1, 3, 8, 8, 8)  # Example input
    noise_labels = torch.randn([1])  # Example noise labels

    out = model(x, noise_labels)
    print(out)

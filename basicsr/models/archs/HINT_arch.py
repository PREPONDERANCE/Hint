## HINT
import math
import numbers
import jittor as jt

from typing import Union, List, Tuple

from jittor import nn
from jittor.einops.einops import rearrange


##########################################################################
## Common Modules


def to_3d(x: jt.Var) -> jt.Var:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: jt.Var, h: int, w: int) -> jt.Var:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def corrcoef(x: jt.Var) -> jt.Var:
    if x.ndim == 1:
        x = jt.unsqueeze(x, dim=0)
    mean_x = x.mean(dim=1, keepdims=True)
    xm = x - mean_x
    cov = (xm @ xm.transpose(0, 1)) / (x.shape[1] - 1)
    d = jt.sqrt(jt.diag(cov))
    corr = cov / (d.unsqueeze(0) * d.unsqueeze(1))
    return corr


class PixelUnshuffle(nn.Module):
    """
    Similar to `torch.nn.PixelUnshuffle`
    """

    def __init__(self, downscale_factor: int):
        super().__init__()

        self.rate = downscale_factor

    def __call__(self, x: jt.Var) -> jt.Var:
        return super().__call__(x)

    def execute(self, x: jt.Var) -> jt.Var:
        r = self.rate
        b, c, h, w = x.shape
        x = (
            jt.reshape(x, (b, c, h // r, r, w // r, r))
            .transpose((0, 1, 3, 5, 2, 4))
            .reshape((b, -1, h // r, w // r))
        )
        return x


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_shape: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(output_shape, int):
            output_shape = (output_shape, output_shape)
        self.output_shape = output_shape

    def __call__(self, x: jt.Var) -> jt.Var:
        return super().__call__(x)

    def execute(self, x: jt.Var) -> jt.Var:
        _is_3d = x.ndim == 3
        if _is_3d:
            x = jt.unsqueeze(x, 0)

        N, C, H_in, W_in = x.shape
        H_out, W_out = self.output_shape

        out = jt.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for i in range(H_out):
            h_start = math.floor(i * H_in / H_out)
            h_end = math.ceil((i + 1) * H_in / H_out)

            for j in range(W_out):
                w_start = math.floor(j * W_in / W_out)
                w_end = math.ceil((j + 1) * W_in / W_out)

                region = x[:, :, h_start:h_end, w_start:w_end]
                if math.prod(region.shape) == 0:
                    h_closest = min(max(int(round(i * H_in / H_out)), 0), H_in - 1)
                    w_closest = min(max(int(round(j * W_in / W_out)), 0), W_in - 1)
                    out[:, :, i, j] = x[:, :, h_closest, w_closest]
                else:
                    out[:, :, i, j] = region.mean(dims=(2, 3))

        return out.squeeze(0) if _is_3d else out


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[numbers.Integral, jt.Var]):
        super(BiasFree_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = jt.size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def execute(self, x: jt.Var) -> jt.Var:
        sigma = jt.var(x, -1, keepdims=True, unbiased=False)
        return x / jt.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[numbers.Integral, jt.Var]):
        super(WithBias_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape = jt.size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.bias = nn.Parameter(jt.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def execute(self, x: jt.Var) -> jt.Var:
        mu = x.mean(-1, keepdims=True)
        sigma = jt.var(x, -1, keepdims=True, unbiased=False)
        return (x - mu) / jt.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, ln_type: str):
        super(LayerNorm, self).__init__()

        self.body = (
            BiasFree_LayerNorm(dim) if ln_type == "BiasFree" else WithBias_LayerNorm(dim)
        )

    def execute(self, x: jt.Var) -> jt.Var:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias,
        )

        self.dwconv = nn.Conv(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.project_in(x)
        dw_out = self.dwconv(x)
        x1, x2 = jt.chunk(dw_out, 2, dim=1)
        x = nn.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c: int = 3):
        super(Inter_CacheModulation, self).__init__()

        self.align = AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(
            in_channels=in_c,
            out_channels=2 * in_c,
            kernel_size=1,
        )
        self.gatingConv = nn.Conv1d(
            in_channels=in_c,
            out_channels=in_c,
            kernel_size=1,
        )

    def execute(self, x1: jt.Var, x2: jt.Var) -> jt.Var:
        x2_pW = self.conv_width(self.align(x2) + x1)
        scale, shift = jt.chunk(x2_pW, 2, dim=1)
        x1_p = x1 * scale + shift
        x1_p = x1_p * nn.gelu(self.gatingConv(x1_p))
        return x1_p


class Intra_CacheModulation(nn.Module):
    def __init__(self, embed_dim: int = 48):
        super(Intra_CacheModulation, self).__init__()

        self.down = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
        )

    def execute(self, x1: jt.Var, x2: jt.Var):
        x_gated = nn.gelu(self.gatingConv(x2 + x1)) * (x2 + x1)
        x_p = self.up(self.down(x_gated))
        return x_p


class ReGroup(nn.Module):
    def __init__(self, groups: List[int] = [1, 1, 2, 4]):
        super(ReGroup, self).__init__()
        self.gourps = groups

    def __call__(
        self, query: jt.Var, key: jt.Var, value: jt.Var
    ) -> Tuple[List[jt.Var], List[jt.Var], List[jt.Var]]:
        return super().__call__(query, key, value)

    def execute(
        self, query: jt.Var, key: jt.Var, value: jt.Var
    ) -> Tuple[List[jt.Var], List[jt.Var], List[jt.Var]]:
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = corrcoef(channel_features)

        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = jt.sort(mean_similarity, descending=True)

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.gourps)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.gourps]

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx

        return query_groups, key_groups, value_groups


def calculate_layer_cache(
    x: List[jt.Var],
    dim: int = 128,
    groups: List[int] = [1, 1, 2, 4],
) -> jt.Var:
    lens = len(groups)
    ceil_dim = dim  # * max_value // sum_value

    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f = jt.mean(qv_cache_f, dim=0, keepdims=True).detach()
        update_elements = nn.interpolate(
            jt.unsqueeze(qv_cache_f, dim=1),
            size=(ceil_dim, ceil_dim),
            mode="bilinear",
            align_corners=False,
        )
        c_i = qv_cache_f.shape[-1]

        if i == 0:
            qv_cache = update_elements * c_i // dim
        else:
            qv_cache = qv_cache + update_elements * c_i // dim

    return qv_cache.squeeze(1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(jt.ones(4, 1, 1))

        self.qkv = nn.Conv(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.group = [1, 2, 2, 3]

        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

        self.inter_modulator1 = Inter_CacheModulation(in_c=1 * dim // 8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3 * dim // 8)
        self.inter_modulators = [
            self.inter_modulator1,
            self.inter_modulator2,
            self.inter_modulator3,
            self.inter_modulator4,
        ]

        self.regroup = ReGroup(self.group)
        self.dim = dim

    def execute(self, x: jt.Var, qv_cache: jt.Var = None) -> Tuple[jt.Var, jt.Var]:
        _, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = jt.chunk(qkv, 3, dim=1)

        q = rearrange(q, "b c h w -> b c (h w)")
        k = rearrange(k, "b c h w -> b c (h w)")
        v = rearrange(v, "b c h w -> b c (h w)")
        qu, ke, va = self.regroup(q, k, v)

        att_score, tmp_cache = [], []
        for index in range(len(self.group)):
            query_head = qu[index]
            key_head = ke[index]

            query_head: jt.Var = jt.normalize(query_head, dim=-1)
            key_head: jt.Var = jt.normalize(key_head, dim=-1)

            attn = jt.matmul(query_head, key_head.transpose(-2, -1))
            attn *= self.temperature[index, :, :]
            attn = nn.softmax(attn, dim=-1)
            att_score.append(attn)  # CxC

            t_cache = query_head.clone().detach() + key_head.clone().detach()
            tmp_cache.append(t_cache)

        tmp_caches = jt.concat(tmp_cache, 1)
        # Inter Modulation
        out = []
        if qv_cache is not None:
            if qv_cache.shape[-1] != c:
                pool = AdaptiveAvgPool2d(c)
                qv_cache = pool(qv_cache)
        for i in range(4):
            if qv_cache is not None:
                inter_modulator = self.inter_modulators[i]
                att_score[i] = inter_modulator(att_score[i], qv_cache) + att_score[i]
                out.append(jt.matmul(att_score[i], va[i]))
            else:
                out.append(jt.matmul(att_score[i], va[i]))

        update_factor = 0.9
        if qv_cache is not None:
            update_elements = calculate_layer_cache(att_score, c, self.group)
            qv_cache = qv_cache * update_factor + update_elements * (1 - update_factor)
        else:
            qv_cache = calculate_layer_cache(att_score, c, self.group)
            qv_cache = qv_cache * update_factor

        out_all = jt.concat(out, dim=1)
        # Intra Modulation
        out_all = self.intra_modulator(out_all, tmp_caches) + out_all

        out_all = rearrange(out_all, "b  c (h w) -> b c h w", h=h, w=w)
        out_all = self.project_out(out_all)
        return (out_all, qv_cache)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        ln_type: str,
        is_att: bool,
    ):
        super(TransformerBlock, self).__init__()
        self.is_att = is_att
        if self.is_att:
            self.norm1 = LayerNorm(dim, ln_type)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, ln_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def execute(self, inputs: Tuple[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        x, qv_cache = inputs

        if self.is_att:
            x_tmp = x
            x_att, qv_cache = self.attn(self.norm1(x), qv_cache=qv_cache)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))
        return (x, qv_cache)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv(
            in_c,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv(
                n_feat,
                n_feat // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            PixelUnshuffle(2),
        )

    def execute(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.body(x)


class HINT(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        ln_type="WithBias",
        dual_pixel_task=False,
        qv_cache=None,
    ):
        super(HINT, self).__init__()

        self.qv_cache = qv_cache
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=False,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=False,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=False,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=True,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=True,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=True,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=True,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    is_att=True,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv(
                dim,
                int(dim * 2**1),
                kernel_size=1,
                bias=bias,
            )

        self.output = nn.Conv(
            int(dim * 2**1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def execute(self, inp_img: jt.Var) -> jt.Var:
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1, self.qv_cache = self.encoder_level1(
            (inp_enc_level1, self.qv_cache)
        )
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, self.qv_cache = self.encoder_level2(
            (inp_enc_level2, self.qv_cache)
        )

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, self.qv_cache = self.encoder_level3(
            (inp_enc_level3, self.qv_cache)
        )

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, self.qv_cache = self.latent((inp_enc_level4, self.qv_cache))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = jt.concat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, self.qv_cache = self.decoder_level3(
            (inp_dec_level3, self.qv_cache)
        )

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = jt.concat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, self.qv_cache = self.decoder_level2(
            (inp_dec_level2, self.qv_cache)
        )

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = jt.concat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, self.qv_cache = self.decoder_level1(
            (inp_dec_level1, self.qv_cache)
        )

        out_dec_level1, self.qv_cache = self.refinement((out_dec_level1, self.qv_cache))

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

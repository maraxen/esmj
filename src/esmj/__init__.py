from dataclasses import fields
from functools import partial, singledispatch

import einops
import equinox as eqx
import esm
import esm.layers
import esm.layers.attention
import esm.layers.blocks
import esm.layers.transformer_stack
import esm.models
import esm.models.esmc
import jax
import numpy as np
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


def convert_tensor(x: torch.Tensor):
    x = x.detach()
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)

    if x.device.type == "cuda":
        x = x.cpu()
    return jnp.array(x)


# basic types
from_torch.register(torch.Tensor, convert_tensor)
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.GELU, lambda _: jax.nn.gelu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])


class AbstractFromTorch(eqx.Module):
    """
    Default implementation of `from_torch` for equinox modules.
    This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
    Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.

    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module):
        # assemble arguments to `cls` constructor from `model`

        field_to_type = {field.name: field.type for field in fields(cls)}
        kwargs = {
            child: from_torch(child_module) for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        }

        # add fields that are not child_modules or parameters
        for field_name, field_type in field_to_type.items():
            if not hasattr(model, field_name):
                if not isinstance(None, field_type):
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                    )
                else:
                    kwargs[field_name] = None
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        # check we're not passing any additional properties
        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
            )

        return cls(**kwargs)


def register_from_torch(torch_module_type):
    """Class decorator to register an equinox module for conversion from a torch module."""

    def decorator(cls):
        from_torch.register(torch_module_type, cls.from_torch)
        return cls

    return decorator


# this isn't very jax-y
def _vmap(f, tensor, *args):
    for _ in range(len(tensor.shape) - 1):
        f = jax.vmap(f)
    return f(tensor, *args)


def vmap_to_last_dimension(f):
    return partial(_vmap, f)


@register_from_torch(torch.nn.Linear)
class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics"""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(l: torch.nn.Linear):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))


@register_from_torch(torch.nn.LayerNorm)
class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics"""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        ln = eqx.nn.LayerNorm(
            shape=x.shape[-1],
            eps=self.eps,
            use_weight=self.weight is not None,
            use_bias=self.bias is not None,
        )
        ln = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            ln,
            (self.weight, self.bias),
            is_leaf=lambda x: x is None,
        )

        return vmap_to_last_dimension(ln)(x)

    @staticmethod
    def from_torch(l: torch.nn.LayerNorm):
        return LayerNorm(weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps)


@register_from_torch(torch.nn.Sequential)
class Sequential(eqx.Module):
    _modules: dict[
        str, AbstractFromTorch
    ]  # IMHO this is a fairly wild design choice, but this is really how pytorch works.

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))


@register_from_torch(torch.nn.modules.sparse.Embedding)
class SparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __call__(self, indices):
        ndims = len(indices.shape)

        def apply(index):
            return self.embedding(index)

        f = apply
        for _ in range(ndims):
            f = jax.vmap(f)

        return f(indices)

    @staticmethod
    def from_torch(m: torch.nn.modules.sparse.Embedding):
        return SparseEmbedding(embedding=eqx.nn.Embedding(weight=from_torch(m.weight)))


@from_torch.register(esm.layers.blocks.SwiGLU)
def _handle(_):
    def swiglu(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(x1) * x2

    return swiglu


@register_from_torch(esm.layers.rotary.RotaryEmbedding)
class RotaryEmbedding(AbstractFromTorch):
    dim: int
    base: int = 10000

    def __call__(self, q: Float[Array, "B N H D"], k: Float[Array, "B N H D"]):
        N = q.shape[1]
        t = jnp.arange(N, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inverse_freq)
        cos = jnp.cos(freqs)[:N]
        sin = jnp.sin(freqs)[:N]

        return (
            self.apply_rotary_emb(q, cos, sin),
            self.apply_rotary_emb(k, cos, sin),
        )

    @property
    def inverse_freq(self):
        return 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))

    @staticmethod
    def rotate_half(x: Float[Array, "B N H D"]):
        x1, x2 = np.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    @staticmethod
    def apply_rotary_emb(
        x: Float[Array, "B N H D"], cos: Float[Array, "N P"], sin: Float[Array, "N P"]
    ):
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        seqlen = x.shape[1]  # x.size(1)
        cos = cos[:seqlen]
        sin = sin[:seqlen]
        cos = einops.repeat(cos, "s d -> s 1 (2 d)")
        sin = einops.repeat(sin, "s d -> s 1 (2 d)")
        return jnp.concatenate(
            [
                x[..., :ro_dim] * cos + RotaryEmbedding.rotate_half(x[..., :ro_dim]) * sin,
                x[..., ro_dim:],
            ],
            axis=-1,
        )


@register_from_torch(esm.layers.attention.MultiHeadAttention)
class MultiHeadAttention(AbstractFromTorch):
    d_model: int
    n_heads: int
    d_head: int
    layernorm_qkv: Sequential
    out_proj: Linear
    rotary: RotaryEmbedding
    q_ln: LayerNorm
    k_ln: LayerNorm

    def _apply_rotary(self, q, k):
        q = einops.rearrange(q, "... (h d) -> ... h d", h=self.n_heads, d=self.d_head)
        k = einops.rearrange(k, "... (h d) -> ... h d", h=self.n_heads, d=self.d_head)
        q, k = self.rotary(q, k)
        q = einops.rearrange(q, "... h d -> ... (h d)", h=self.n_heads, d=self.d_head)
        k = einops.rearrange(k, "... h d -> ... (h d)", h=self.n_heads, d=self.d_head)
        return q, k

    def __call__(self, x):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = jnp.split(qkv_BLD3, 3, axis=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD),
            self.k_ln(key_BLD),
        )

        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        query_BHLD, key_BHLD, value_BHLD = map(
            lambda x: einops.rearrange(x, pattern="b s (h d) -> b h s d", h=self.n_heads),
            (query_BLD, key_BLD, value_BLD),
        )

        context_BHLD = jax.nn.dot_product_attention(
            einops.rearrange(query_BHLD, "B H S D -> B S H D"),
            einops.rearrange(key_BHLD, "B H S D -> B S H D"),
            einops.rearrange(value_BHLD, "B H S D -> B S H D"),
        )

        context_BHLD = einops.rearrange(context_BHLD, "b s h d -> b s (h d)")
        return self.out_proj(context_BHLD)


@register_from_torch(esm.layers.blocks.UnifiedTransformerBlock)
class UnifiedTransformerBlock(AbstractFromTorch):
    ffn: Sequential
    attn: MultiHeadAttention
    scaling_factor: float

    def __call__(self, x):
        x = x + self.attn(x) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


@register_from_torch(esm.layers.transformer_stack.TransformerStack)
class TransformerStack(AbstractFromTorch):
    block_params: UnifiedTransformerBlock
    block_static: UnifiedTransformerBlock
    norm: LayerNorm

    def __call__(self, x: Float[Array, "B N D"]):
        def body(x, params):
            layer = eqx.combine(self.block_static, params)
            x = layer(x)
            return x, x

        final_state, all_states = jax.lax.scan(
            body,
            x,
            self.block_params,
        )
        return self.norm(final_state), all_states

    @staticmethod
    def from_torch(m: esm.layers.transformer_stack.TransformerStack):
        blocks = [from_torch(b) for b in m.blocks]
        block_params = jax.tree.map(
            lambda *v: jnp.stack(v),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        block_static = eqx.partition(blocks[0], eqx.is_inexact_array)[1]
        return TransformerStack(
            block_params=block_params,
            block_static=block_static,
            norm=from_torch(m.norm),
        )


class ESMCOutput(eqx.Module):
    logits: Float[Array, "B N V"]
    embedding: Float[Array, "B N D"]
    hiddens: Float[Array, "B N L D"]


@register_from_torch(esm.models.esmc.ESMC)
class ESMC(eqx.Module):
    embed: SparseEmbedding
    transformer: TransformerStack
    sequence_head: Sequential

    def __call__(self, tokens: Int[Array, "B N"]):
        assert tokens.ndim == 2, f"Expected 2D input, got {tokens.ndim}D"
        x = self.embed(tokens)
        x, hiddens = self.transformer(x)
        logits = self.sequence_head(x)
        return ESMCOutput(
            logits=logits,
            embedding=x,
            hiddens=hiddens,
        )

    @staticmethod
    def from_torch(m: esm.models.esmc.ESMC):
        return ESMC(
            embed=from_torch(m.embed),
            transformer=from_torch(m.transformer),
            sequence_head=from_torch(m.sequence_head),
        )

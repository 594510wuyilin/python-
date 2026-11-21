"""NumPy implementation of the OpenCLIP ViT-B/32 image encoder.

Implements patch embedding, multi-head self-attention, Transformer residual
blocks, and final CLS readout/projection using supplied float32 weights.
Retains the public class interfaces for reuse in the final multimodal project.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']   # [3D, D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']       # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D, D]
        self.out_bias = weights[prefix + '.out_proj.bias']      # [D]
        self.heads = heads
        self.width = self.out_bias.shape[0]
        if self.width % heads != 0:
            raise ValueError(f'Embedding width {self.width} must be divisible by heads {heads}.')
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """Split feature dimension into multiple attention heads.

        Input:  [B, N, D]
        Output: [B, H, N, D/H] (preserves token order along axis 2)
        """
        batch, tokens, width = x.shape
        if width != self.width:
            raise ValueError(f'Expected feature width {self.width}, got {width}.')
        heads = x.reshape(batch, tokens, self.heads, self.head_dim)
        return heads.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Stored linear weights follow [out, in] layout.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2 * self.width])
        v = self.split_heads(qkv[:, :, 2 * self.width:])

        # Scaled dot-product attention scores: [B, H, N, N] (keep float32)
        scale = np.sqrt(np.float32(self.head_dim))
        scores = (q @ k.swapaxes(-1, -2)) / scale

        # Optional causal mask for text-encoder compatibility (images use causal=False).
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)

        # Weighted value aggregation and head merging: [B, H, N, D/H] -> [B, N, D]
        context = coefficients @ v
        batch, tokens, _ = x.shape
        merged = context.transpose(0, 2, 1, 3).reshape(batch, tokens, self.width)

        # Output linear projection: [B, N, D]
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix + '.ln_1.weight'], weights[prefix + '.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix + '.ln_2.weight'], weights[prefix + '.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix + '.attn', heads)
        self.fc1_weight = weights[prefix + '.mlp.c_fc.weight']   # [4D, D]
        self.fc1_bias = weights[prefix + '.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix + '.mlp.c_proj.weight'] # [D, 4D]
        self.fc2_bias = weights[prefix + '.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Pre-LN Multi-Head Self-Attention with first residual connection
        update = self.attention(self.norm1(x), causal=causal)
        residual = x + update

        # Pre-LN two-layer GELU MLP with second residual connection
        u = self.norm2(residual)
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50, 768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768, 512], already [in, out]

    def patchify(self, images):
        """Split preprocessed image batch into flattened 32x32 RGB patches.

        Input:  [B, 3, 224, 224]
        Output: [B, 49, 3072] (patches ordered left-to-right, top-to-bottom; C,H,W inside)
        """
        if images.ndim != 4:
            raise ValueError(f'Expected 4D image batch [B, C, H, W], got shape {images.shape}.')
        batch, channels, height, width = images.shape
        patch_size = 32
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f'Image spatial size ({height}, {width}) must be divisible by {patch_size}.')
        rows, cols = height // patch_size, width // patch_size
        grid = images.reshape(batch, channels, rows, patch_size, cols, patch_size)
        patches = grid.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, rows * cols, channels * patch_size ** 2)

    def embed(self, images):
        """Project patches, prepend CLS token, and add learned positional embeddings."""
        patches = self.patchify(images)  # [B, 49, 3072]
        patch_tokens = patches @ self.patch_weight.T  # [B, 49, 768], no bias

        cls = np.tile(self.cls, (images.shape[0], 1, 1))
        tokens = np.concatenate((cls, patch_tokens), axis=1)  # [B, 50, 768]
        return tokens + self.position

    def encode_tokens(self, images):
        """Run Pre-Norm and all 12 Transformer blocks; returns [B, 50, 768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """Encode image batch into L2-normalized [B, 512] float32 feature vectors."""
        tokens = self.encode_tokens(images)

        # Select CLS token (row 0), apply post-Norm, project via [768, 512] matrix
        cls = tokens[:, 0, :]
        normalized_cls = self.post_norm(cls)
        output = normalized_cls @ self.projection

        return unit_vector(output)

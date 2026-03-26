"""TensorFlow BERT-style model for genotype token classification.

Pipeline summary
----------------
1. Build a masked view of the full genotype dataset by masking tokens whose
   `var_id` is NOT present in a given array-SNP list.
2. Build a compact unmasked subset that only keeps overlapping variants
   (genotype + aligned position + block + var_id).
3. Encode the compact subset with a BERT-style encoder to obtain contextual
   embeddings used as K/V memory for cross-attention.
4. Encode the full masked dataset with token/position/block embeddings,
   cross-attend to subset memory, then run additional transformer layers.
5. Produce per-token binary classification logits (class 0/1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import tensorflow as tf


@dataclass
class GenotypeBatch:
    """Container for one minibatch of genotype sequence data."""

    genotypes: tf.Tensor  # [batch, seq_len], int ids
    positions: tf.Tensor  # [batch, seq_len], int ids/indices
    blocks: tf.Tensor  # [batch, seq_len], int ids
    var_ids: tf.Tensor  # [batch, seq_len], int ids
    labels: Optional[tf.Tensor] = None  # [batch, seq_len], 0/1 for token cls


@dataclass
class PreparedInputs:
    """Two prepared datasets for dual-stream model input."""

    full_masked: Dict[str, tf.Tensor]
    subset_unmasked: Dict[str, tf.Tensor]
    subset_valid_mask: tf.Tensor


def prepare_dual_inputs(
    batch: GenotypeBatch,
    array_snp_ids: tf.Tensor,
    genotype_mask_token_id: int,
    pad_token_id: int = 0,
) -> PreparedInputs:
    """Create full masked + unmasked subset inputs.

    Args:
        batch: Original full dataset tensors [B, L].
        array_snp_ids: 1D tensor of allowed SNP var IDs [N].
        genotype_mask_token_id: Token id used when variant is not in SNP array.
        pad_token_id: Padding token for subset tensors.

    Returns:
        PreparedInputs:
          - full_masked: full sequence with non-overlap genotypes masked.
          - subset_unmasked: compact sequence with only overlap variants.
          - subset_valid_mask: [B, L] bool mask indicating valid subset tokens.
    """
    var_in_array = tf.reduce_any(
        tf.equal(batch.var_ids[..., tf.newaxis], array_snp_ids[tf.newaxis, tf.newaxis, :]),
        axis=-1,
    )  # [B, L] bool

    # 1) Full dataset with masked genotypes for non-overlapping variants.
    full_genotypes_masked = tf.where(
        var_in_array,
        batch.genotypes,
        tf.fill(tf.shape(batch.genotypes), tf.cast(genotype_mask_token_id, batch.genotypes.dtype)),
    )

    full_masked = {
        "genotypes": full_genotypes_masked,
        "positions": batch.positions,
        "blocks": batch.blocks,
        "var_ids": batch.var_ids,
        "attention_mask": tf.cast(tf.not_equal(full_genotypes_masked, pad_token_id), tf.int32),
    }

    # 2) Compact subset dataset with only overlapping variants kept.
    #    Non-overlap tokens are replaced with PAD and later valid positions are tracked.
    subset_genotypes = tf.where(
        var_in_array,
        batch.genotypes,
        tf.fill(tf.shape(batch.genotypes), tf.cast(pad_token_id, batch.genotypes.dtype)),
    )
    subset_positions = tf.where(
        var_in_array,
        batch.positions,
        tf.fill(tf.shape(batch.positions), tf.cast(0, batch.positions.dtype)),
    )
    subset_blocks = tf.where(
        var_in_array,
        batch.blocks,
        tf.fill(tf.shape(batch.blocks), tf.cast(0, batch.blocks.dtype)),
    )
    subset_var_ids = tf.where(
        var_in_array,
        batch.var_ids,
        tf.fill(tf.shape(batch.var_ids), tf.cast(0, batch.var_ids.dtype)),
    )

    subset_unmasked = {
        "genotypes": subset_genotypes,
        "positions": subset_positions,
        "blocks": subset_blocks,
        "var_ids": subset_var_ids,
        "attention_mask": tf.cast(var_in_array, tf.int32),
    }

    return PreparedInputs(
        full_masked=full_masked,
        subset_unmasked=subset_unmasked,
        subset_valid_mask=var_in_array,
    )


class TransformerBlock(tf.keras.layers.Layer):
    """Standard Transformer encoder block."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, attention_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Keras MHA supports attention_mask shape [B, T] for key masking.
        attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


class CrossAttentionBlock(tf.keras.layers.Layer):
    """Cross-attention where query attends to memory key/value."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, dropout=dropout
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        query_states: tf.Tensor,
        memory_states: tf.Tensor,
        memory_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # memory_mask: [B, S_mem] where 1 means valid.
        out = self.cross_attn(
            query=query_states,
            key=memory_states,
            value=memory_states,
            attention_mask=memory_mask,
            training=training,
        )
        return self.norm(query_states + self.drop(out, training=training))


class GenotypeBertCrossModel(tf.keras.Model):
    """BERT-style dual-stream model with cross-attention for token classification."""

    def __init__(
        self,
        genotype_vocab_size: int,
        position_vocab_size: int,
        block_vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        subset_encoder_layers: int = 2,
        main_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared embedding idea for both streams.
        self.geno_emb = tf.keras.layers.Embedding(genotype_vocab_size, hidden_dim)
        self.pos_emb = tf.keras.layers.Embedding(position_vocab_size, hidden_dim)
        self.block_emb = tf.keras.layers.Embedding(block_vocab_size, hidden_dim)
        self.emb_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.emb_drop = tf.keras.layers.Dropout(dropout)

        self.subset_encoders = [
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(subset_encoder_layers)
        ]

        self.cross_attention = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        self.main_encoders = [
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(main_encoder_layers)
        ]

        # Per-token binary classification logits.
        self.classifier = tf.keras.layers.Dense(2)

    def embed_triplet(
        self,
        genotypes: tf.Tensor,
        positions: tf.Tensor,
        blocks: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        x = self.geno_emb(genotypes) + self.pos_emb(positions) + self.block_emb(blocks)
        x = self.emb_norm(x)
        return self.emb_drop(x, training=training)

    def call(
        self,
        full_masked: Dict[str, tf.Tensor],
        subset_unmasked: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        # Stream 1: subset encoder (memory for cross attention).
        subset_x = self.embed_triplet(
            subset_unmasked["genotypes"],
            subset_unmasked["positions"],
            subset_unmasked["blocks"],
            training=training,
        )
        subset_mask = subset_unmasked["attention_mask"]
        for layer in self.subset_encoders:
            subset_x = layer(subset_x, attention_mask=subset_mask, training=training)

        # Stream 2: full masked sequence encoder with cross attention to subset memory.
        full_x = self.embed_triplet(
            full_masked["genotypes"],
            full_masked["positions"],
            full_masked["blocks"],
            training=training,
        )
        full_mask = full_masked["attention_mask"]

        # Cross-attention: Q from full stream, K/V from subset stream.
        full_x = self.cross_attention(
            query_states=full_x,
            memory_states=subset_x,
            memory_mask=subset_mask,
            training=training,
        )

        for layer in self.main_encoders:
            full_x = layer(full_x, attention_mask=full_mask, training=training)

        return self.classifier(full_x)  # [B, L, 2]


def build_train_step(model: GenotypeBertCrossModel, learning_rate: float = 1e-4):
    """Create a tf.function training step for token-level 0/1 classification."""

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    @tf.function
    def train_step(
        batch: GenotypeBatch,
        array_snp_ids: tf.Tensor,
        genotype_mask_token_id: int,
    ):
        prepared = prepare_dual_inputs(
            batch=batch,
            array_snp_ids=array_snp_ids,
            genotype_mask_token_id=genotype_mask_token_id,
            pad_token_id=0,
        )

        if batch.labels is None:
            raise ValueError("batch.labels is required for training.")

        with tf.GradientTape() as tape:
            logits = model(
                full_masked=prepared.full_masked,
                subset_unmasked=prepared.subset_unmasked,
                training=True,
            )  # [B, L, 2]

            token_loss = loss_fn(batch.labels, logits)  # [B, L]
            valid_mask = tf.cast(prepared.full_masked["attention_mask"], token_loss.dtype)
            loss = tf.reduce_sum(token_loss * valid_mask) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        preds = tf.argmax(logits, axis=-1, output_type=batch.labels.dtype)
        correct = tf.cast(tf.equal(preds, batch.labels), tf.float32) * tf.cast(valid_mask, tf.float32)
        acc = tf.reduce_sum(correct) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)

        return {"loss": loss, "accuracy": acc}

    return train_step


if __name__ == "__main__":
    # Minimal runnable example with synthetic shapes.
    batch_size, seq_len = 4, 16
    vocab_sizes = {
        "genotype_vocab_size": 6,   # e.g. PAD/REF/ALT/HET/MASK/etc.
        "position_vocab_size": 2000,
        "block_vocab_size": 128,
    }

    sample_batch = GenotypeBatch(
        genotypes=tf.random.uniform([batch_size, seq_len], minval=1, maxval=5, dtype=tf.int32),
        positions=tf.random.uniform([batch_size, seq_len], minval=1, maxval=512, dtype=tf.int32),
        blocks=tf.random.uniform([batch_size, seq_len], minval=1, maxval=32, dtype=tf.int32),
        var_ids=tf.random.uniform([batch_size, seq_len], minval=100, maxval=200, dtype=tf.int32),
        labels=tf.random.uniform([batch_size, seq_len], minval=0, maxval=2, dtype=tf.int32),
    )

    array_snp_ids = tf.constant([101, 103, 110, 150, 175], dtype=tf.int32)
    mask_token_id = 5

    model = GenotypeBertCrossModel(**vocab_sizes)
    step_fn = build_train_step(model)
    metrics = step_fn(sample_batch, array_snp_ids=array_snp_ids, genotype_mask_token_id=mask_token_id)
    tf.print("train metrics:", metrics)

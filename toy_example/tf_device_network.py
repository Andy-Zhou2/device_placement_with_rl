from tf_device_measure import OP_TYPES, ADJ_FORWARD, ADJ_BACKWARD, SHAPES, NUM_DEVICE_CHOICES
import torch
import torch.nn as nn
from torch.distributions import Categorical

op_types_tensor = torch.tensor(OP_TYPES, dtype=torch.long)  # shape [3]
adj_forward_tensor = torch.tensor(ADJ_FORWARD, dtype=torch.float32)  # shape [3,3]
adj_backward_tensor = torch.tensor(ADJ_BACKWARD, dtype=torch.float32)  # shape [3,3]
shape_tensor = torch.tensor(SHAPES, dtype=torch.float32)  # shape [3,4]


class OperationEmbeddingNet(nn.Module):
    """
    Builds an embedding for each operation.
    We'll produce a single vector that is a concatenation of:
      - type_embedding (via nn.Embedding)
      - shape_embedding (via a small MLP)
      - adjacency_embedding (via a small MLP)
    Then the final dimension is something like embed_dim.
    """

    def __init__(self, type_vocab_size=2, embed_dim=32):
        super().__init__()
        self.type_embed_dim = 8
        self.shape_embed_dim = 8
        self.out_dim = embed_dim

        # Type embedding
        self.type_embedding = nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_dim=self.type_embed_dim
        )

        # We'll flatten shape (4D) and map to shape_embed_dim
        self.shape_mlp = nn.Sequential(
            nn.Linear(4, self.shape_embed_dim),
            nn.ReLU(),
        )

        # Final linear to unify them
        final_in_dim = self.type_embed_dim + self.shape_embed_dim + 3 + 3  # 3 for adj
        self.final_linear = nn.Linear(final_in_dim, self.out_dim)

    def forward(self, op_types, op_shapes, op_adj_forward, op_adj_backward):
        """
        Each of these is shape [3, ...] if we have 3 operations.
         - op_types: [3]
         - op_shapes: [3, 4]
         - op_adjs: [3, 3]
        We'll produce [3, embed_dim].
        """
        # Type embedding: shape [3, type_embed_dim]
        t_emb = self.type_embedding(op_types)

        # shape embedding: [3, shape_embed_dim]
        s_emb = self.shape_mlp(op_shapes)

        # Concat
        cat_emb = torch.cat([t_emb, s_emb, op_adj_forward, op_adj_backward], dim=-1)  # [3, final_in_dim]

        out = self.final_linear(cat_emb)  # [3, embed_dim]
        return out


class AutoRegressiveTransformerPolicy(nn.Module):
    """
    We have 3 operations: [X, dense1, dense2].
    We'll produce 3 device tokens in an auto-regressive manner:
      token[0] -> device for X
      token[1] -> device for dense1
      token[2] -> device for dense2
    The input to the transformer includes the operation embedding +
    the previously chosen device tokens (if any).
    """

    def __init__(self, embed_dim=32, n_heads=2, n_layers=2):
        super().__init__()
        self.op_embedder = OperationEmbeddingNet(type_vocab_size=2, embed_dim=embed_dim)

        # We'll have an embedding for chosen device tokens.
        self.dev_token_embed = nn.Embedding(NUM_DEVICE_CHOICES, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=64,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # We'll produce logits for each step (one device selection).
        # We do so by reading the transformer's last hidden for that step.
        self.token_heads = nn.Linear(embed_dim, NUM_DEVICE_CHOICES)

        self.op_types_tensor = op_types_tensor  # shape [3]
        self.adj_forward_tensor = adj_forward_tensor  # shape [3,3]
        self.adj_backward_tensor = adj_backward_tensor  # shape [3,3]
        self.shape_tensor = shape_tensor  # shape [3,4]

    def forward(self):
        """
        If causal=True, we apply a causal mask so each operation (token)
        only sees the previous ops in self-attention.

        Return: a list of 3 logits: [logits0, logits1, logits2]
          - each shaped [B, NUM_DEVICE_CHOICES]
        """
        batch_size = 1

        op_emb = self.op_embedder(
            self.op_types_tensor,  # shape [3]
            self.shape_tensor,  # shape [3,4]
            self.adj_forward_tensor,  # shape [3,3]
            self.adj_backward_tensor,  # shape [3,3]
        )  # => [3, embed_dim]

        op_emb = op_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # e.g. shape => [B, 3, embed_dim]

        out = self.transformer(op_emb)  # => [B, 3, embed_dim]

        logits_list = self.token_heads(out)  # => [B, 3, NUM_DEVICE_CHOICES]

        return logits_list

    def sample_action_and_logprob(self):
        """
        Returns:
            action (torch.LongTensor): shape [batch_size, m]
            log_prob (torch.Tensor):   shape [batch_size]
        """
        logits = self.forward()  # [batch_size, m, n]
        dists = Categorical(logits=logits)

        action = dists.sample()
        log_prob = dists.log_prob(action).sum(dim=-1)
        return action, log_prob, logits

    def get_log_prob(self, action):
        """
        Given a specific action, return the log-probability.
        """
        logits = self.forward()
        dists = Categorical(logits=logits)
        log_prob = dists.log_prob(action).sum(dim=-1)
        return log_prob, logits

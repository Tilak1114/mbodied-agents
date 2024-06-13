from typing import Dict

import torch
import torch.nn as nn
from gym import spaces
from mbodied_agents.agents.motion.rt1.tokenizers.action_tokenizer import RT1ActionTokenizer
from mbodied_agents.agents.motion.rt1.tokenizers.image_tokenizer import RT1ImageTokenizer
from mbodied_agents.agents.motion.rt1.transformer import Transformer


class TransformerNetwork(nn.Module):
    """A transformer based actor network."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Dict,
        context_key: str = 'natural_language_embedding',
        vocab_size: int = 256,  # Token dimension.
        num_heads: int = 8,
        image_tokens_size: int = 8,
        token_embedding_dim: int = 512,  # Embedded token dimension.
        num_layers: int = 1,
        layer_size: int = 4096, # Attention key_dim which is the size of each attention head for query, key and values.
        dropout_rate: float = 0.1,
        use_token_learner: bool = True,
        observation_history_length: int = 6,
        future_prediction_length: int = 1,
        causal_attention: bool = True,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_history_length = observation_history_length
        self.future_prediction_length = future_prediction_length
        self.causal_attention = causal_attention
        self.vocab_size = vocab_size
        self.context_key = context_key
        self.num_heads = num_heads
        self.use_token_learner = use_token_learner
        self.image_tokens_size = image_tokens_size

        self.loss_object = nn.CrossEntropyLoss(reduction="none")

        self.image_tokenizers = RT1ImageTokenizer(
            embedding_output_dim=token_embedding_dim,
            language_embedding_size=token_embedding_dim,
            use_token_learner=use_token_learner,
            num_tokens=self.image_tokens_size,
        )

        self.action_tokenizer = RT1ActionTokenizer(
            action_space, vocab_size=self.vocab_size,
        )

        self.transformer = Transformer(
            num_layers=num_layers,
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=token_embedding_dim,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size,
            input_token_emb_dim=token_embedding_dim,
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> tuple:
        """Calls the transformer network.

        Args:
            observations: Observation data including image and natural language
                embedding in dict of Tensors.

        Returns:
            A tuple `(Detokenized output actions, network state)`.
        """
        pass


if __name__ == "__main__":
    net = TransformerNetwork()

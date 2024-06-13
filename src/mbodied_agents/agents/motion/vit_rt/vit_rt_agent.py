# Copyright 2024 Mbodi AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from gym import spaces
from mbodied_agents.agents.motion.motor_agent import MotorAgent
from mbodied_agents.agents.motion.vit_rt.tokenizers.action_tokenizer import RT1ActionTokenizer
from mbodied_agents.agents.motion.vit_rt.tokenizers.utils import batched_space_sampler, np_to_tensor
from mbodied_agents.agents.motion.vit_rt.transformer_network import TransformerNetwork
from mbodied_agents.base.motion import Motion
from mbodied_agents.types.controls import HandControl, JointControl, Pose
from mbodied_agents.types.vision import SupportsImage
from transformers import BertModel, BertTokenizer

observation_space = spaces.Dict(
    {
        "image_primary": spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
        "natural_language_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=[768], dtype=np.float32),
    },
)
action_space = spaces.Dict(
    OrderedDict(
        [
            (
                "xyz",
                spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            ),
            (
                "rpy",
                spaces.Box(low=-np.pi, high=np.pi,
                           shape=(3,), dtype=np.float32),
            ),
            (
                "grasp",
                spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            ),
        ],
    ),
)

IMAGE_HISTORY_BUFFER_SIZE = 6
IMAGE_SIZE = (224, 224, 3)


class ViTRTAgent(MotorAgent):
    """RT1Agent class responsible for interacting with the environment based on the agent's policy.

    This agent uses a TransformerNetwork to generate actions based on the given observations
    and context (natural language embedding).

    Attributes:
        config (dict): Configuration dictionary containing parameters for the agent.
        device (torch.device): The device to run the computations (CPU or CUDA).
        model (TransformerNetwork): The neural network model used for predicting actions.
        policy_state (Optional[torch.Tensor]): The internal state of the policy network.
        action_tokenizer (RT1ActionTokenizer): The action tokenizer for converting output to actions.
        image_history (List[torch.Tensor]): History of the past observations.
        step_num (int): Keeps track of the number of steps taken by the agent.
    """

    def __init__(self, config, **kwargs) -> None:
        """Initializes the RT1Agent with the provided configuration and model weights.

        Args:
            config (dict): Configuration parameters for setting up the agent.
            weights_path (str, optional): Path to the pre-trained model weights. Defaults to None.
            **kwargs: Additional keyword arguments.

        Example:
            config = {
                "observation_history_size": 6,
                "future_prediction": 6,
                "token_embedding_dim": 512,
                "causal_attention": True,
                "num_layers": 6,
                "layer_size": 512,
                "observation_space": observation_space,
                "action_space": action_space,
                "history_size": IMAGE_HISTORY_BUFFER_SIZE,
                "image_size": IMAGE_SIZE
            }
            agent = RT1Agent(config, weights_path="path/to/weights.pth")
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.text_encoder_model = BertModel.from_pretrained(model_name)

        self.model = TransformerNetwork(
            observation_history_length=config["observation_history_size"],
            future_prediction_length=config["future_prediction"],
            token_embedding_dim=config["token_embedding_dim"],
            causal_attention=config["causal_attention"],
            num_layers=config["num_layers"],
            layer_size=config["layer_size"],
            observation_space=config.get(
                "observation_space", observation_space),
            action_space=config.get("action_space", action_space),
            image_keys=["image_primary"],
            context_key="natural_language_embedding",
        ).to(self.device).eval()

        self.policy_state = None
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space,
        )

    def get_text_embedding(self, text):

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt')

        # Get the embeddings
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)

        # The last hidden states are in `outputs.last_hidden_state`
        # You can typically use the embeddings of [CLS] token for the entire sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings

    def act(self,
            instruction: str,
            image_seq: List[SupportsImage],
            ) -> List[Motion]:
        
        images = torch.stack(self.image_history)[None]

        video = rearrange(images.to(self.device), "b f h w c -> b f c h w")
        self.policy_state = np_to_tensor(
            batched_space_sampler(
                self.model.state_space,
                batch_size=1,
            ),
        )

        instruction_emb = torch.tensor(self.get_text_embedding(instruction))

        obs = {
            "image_primary": video,
            "natural_language_embedding": repeat(instruction_emb, "b c -> (6 b) c"),
        }

        outs, network_state = self.model(
            obs,
            self.policy_state,
        )
        out_tokens = outs[:, : (5 + 1), :, :].detach().cpu().argmax(dim=-1)
        self.out_tokens = out_tokens

        self.policy_state = network_state

        outs = self.action_tokenizer.detokenize(out_tokens)
        actions = [
            HandControl(
                pose=Pose(
                    x=outs["xyz"][0][i][0],
                    y=outs["xyz"][0][i][1],
                    z=outs["xyz"][0][i][2],
                    roll=outs["rpy"][0][i][0],
                    pitch=outs["rpy"][0][i][1],
                    yaw=outs["rpy"][0][i][2],
                ),
                grasp=JointControl(value=outs["grasp"][0][i]),
            )
            for i in range(6)
        ]

        self.step_num += 1
        return actions


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
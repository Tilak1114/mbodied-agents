from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from gym import spaces
from mbodied_agents.agents.motion.rt1.tokenizers.action_tokenizer import RT1ActionTokenizer
from mbodied_agents.agents.motion.rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from mbodied_agents.agents.motion.rt1.transformer_network import TransformerNetwork
from mbodied_agents.base.agent import Agent
from mbodied_agents.base.motion import Motion
from mbodied_agents.types.controls import HandControl, JointControl, Pose
from mbodied_agents.types.vision import SupportsImage

observation_space = spaces.Dict(
    {
        "image_primary": spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
        "natural_language_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32),
    },
)
action_space_dict = OrderedDict(
    [
        (
            "xyz",
            spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        ),
        (
            "rpy",
            spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
        ),
        (
            "grasp",
            spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
        ),
    ],
)


class RT1Agent(Agent):
    def __init__(self, config, weights_path: str = None, **kwargs) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerNetwork(
            observation_history_length=config["observation_history_size"],
            future_prediction_length=config["future_prediction"],
            token_embedding_dim=config["token_embedding_dim"],
            causal_attention=config["causal_attention"],
            num_layers=config["num_layers"],
            layer_size=config["layer_size"],
            observation_space=observation_space,
            action_space=spaces.Dict(action_space_dict),
            image_keys=["image_primary"],
            context_key="natural_language_embedding",
        ).to(self.device).eval()

        self.policy_state = None
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space_dict)

        if weights_path:
            self.model.load_state_dict(torch.load(
                weights_path, map_location=self.device))
        self.image_history = []
        for _i in range(6):
            self.image_history.append(torch.zeros(
                size=(224, 224, 3), dtype=torch.float, device=self.device))

        self.step_num: int = 0

    def act(
        self,
        instruction_emb: torch.Tensor,
        image: SupportsImage,
        mean_tokens=None,
        std_tokens=None,
    ) -> List[Motion]:
        image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))
        self.image_history.append(torch.tensor(
            image / 255.0, dtype=torch.float32, device=self.device).permute(1, 0, 2))
        if len(self.image_history) > 6:
            self.image_history.pop(0)
        elif len(self.image_history) < 6:
            for _ in range(6 - len(self.image_history)):
                self.image_history.append(
                    torch.tensor(image / 255.0, dtype=torch.float32,
                                 device=self.device).permute(1, 0, 2),
                )

        images = torch.stack(self.image_history)[None]

        video = rearrange(images.to(self.device), "b f h w c -> b f c h w")
        self.policy_state = np_to_tensor(
            batched_space_sampler(
                self.model.state_space,
                batch_size=1,
            ),
        )

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

        outs = self.action_tokenizer.detokenize(
            out_tokens, action_mean=mean_tokens, action_std=std_tokens)
        actions = [
            HandControl(
                pose=Pose(
                    xyz=np.array(outs["xyz"][0][i]),
                    rpy=np.array(outs["rpy"][0][i]),
                ),
                grasp=JointControl(value=outs["grasp"][0][i]),
            )
            for i in range(6)
        ]

        for _action in actions:
            pass

        self.step_num += 1
        return actions
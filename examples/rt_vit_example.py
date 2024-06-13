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

import torch
from mbodied_agents.agents.motion.vit_rt.vit_rt_agent import ViTRTAgent


def main() -> None:
    # Define the configuration for the RT1Agent
    rt1_agent_config = {
        "num_layers": 8,
        "layer_size": 128,
        "observation_history_size": 6,
        "future_prediction": 6,
        "token_embedding_dim": 768,
        "causal_attention": True,
    }

    # Instantiate the RT1Agent
    rt1_agent = ViTRTAgent(rt1_agent_config)

    # Create dummy input data
    image = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
    instruction = 'Pick the banana peel'

    # Use the act method of RT1Agent to get actions
    actions = rt1_agent.act(image_seq=image, instruction=instruction)

    # Print the actions
    print("Actions received from RT1Agent:")
    print(actions)

if __name__ == "__main__":
    main()

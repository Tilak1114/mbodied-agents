import io

import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Load the Hugging Face dataset non-streaming mode
ds = load_dataset("jxu124/OpenX-Embodiment",
                  "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
                  streaming=False,
                  split='train')

TRAJECTORY_WINDOW_SIZE = 6
FUTURE_ACTION_WINDOW_SIZE = 6


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.cache = []

        for episode in tqdm(self.dataset, desc="Processing episodes"):
            steps = episode['data.pickle']['steps']
            self.process_steps(steps)

    def process_steps(self, steps):
        for si in range(len(steps) - (TRAJECTORY_WINDOW_SIZE + FUTURE_ACTION_WINDOW_SIZE)):
            trajectory_steps = steps[si:si + TRAJECTORY_WINDOW_SIZE]
            future_steps = steps[si + TRAJECTORY_WINDOW_SIZE:si +
                                 TRAJECTORY_WINDOW_SIZE + FUTURE_ACTION_WINDOW_SIZE]

            if len(future_steps) == FUTURE_ACTION_WINDOW_SIZE:
                item_map = {
                    'trajectory_steps': [
                        (self.transform_image(
                            np.array(Image.open(io.BytesIO(
                                step['observation']['image']['bytes'])))
                        ), step['language_instruction']) for step in trajectory_steps
                    ],
                    'future_steps': [
                        step['action'] for step in future_steps
                    ],
                }
                self.cache.append(item_map)

    def transform_image(self, image):
        if self.transform:
            image = Image.fromarray(image)
            return self.transform(image)
        return image

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_data_loader(batch_size):
    custom_dataset = CustomDataset(ds, transform=transform)
    return DataLoader(custom_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

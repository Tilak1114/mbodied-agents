import io
import os

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

# Step 1: Load the Hugging Face dataset in streaming mode
ds = load_dataset("jxu124/OpenX-Embodiment",
                  "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
                  streaming=True,
                  split='train')

TRAJECTORY_WINDOW_SIZE = 6
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
STEPS_PER_EPOCH = 1000
SAVE_PATH = "/home/user/tilak/mbodied-agents/src/mbodied_agents/agents/motion/rt1/clip/checkpoints"

# Step 2: Define a custom PyTorch IterableDataset
class HuggingFaceIterableDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for index, item in enumerate(self.hf_dataset):
            steps = item['data.pickle']['steps']
            if len(steps) < TRAJECTORY_WINDOW_SIZE:
                continue
            for i in range(len(steps) - TRAJECTORY_WINDOW_SIZE):
                trajectory_steps = steps[i:i + TRAJECTORY_WINDOW_SIZE]
                language_instruction = trajectory_steps[0]['language_instruction']
                images_tensor_list = []
                for idx, t_step in enumerate(trajectory_steps):
                    image_bytes = t_step['observation']['image']['bytes']
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    transform = transforms.ToTensor()
                    image_tensor = transform(image)
                    images_tensor_list.append(image_tensor)
                yield torch.stack(images_tensor_list), language_instruction
                
def get_dataloader(ds, batch_size):
    dataset = HuggingFaceIterableDataset(ds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

data_loader = get_dataloader(ds, BATCH_SIZE)

# Define a LightningModule
class CLIPFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=LEARNING_RATE):
        super(CLIPFineTuner, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.learning_rate = learning_rate

    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        images_batch, texts_batch = batch
    
        # Reshape the images_batch for processing
        batch_size, seq_len, C, H, W = images_batch.shape
        images_batch_flatten = images_batch.view(batch_size * seq_len, C, H, W)

        # Duplicate texts_batch to match the sequence length
        texts_batch_flatten = [text for text in texts_batch for _ in range(seq_len)]
        
        # Pass the flattened images and expanded text batch to the model
        outputs = self(images_batch_flatten, texts_batch_flatten)
        
        # Reshape the logits to (batch_size, seq_len, num_logits)
        logits_per_image = outputs.logits_per_image.view(batch_size, seq_len, -1)
        logits_per_text = outputs.logits_per_text.view(batch_size, seq_len, -1)
        
        # Reduce over the sequence dimension
        logits_per_image = logits_per_image.mean(dim=1)
        logits_per_text = logits_per_text.mean(dim=1)
        
        # Compute the contrastive loss
        labels = torch.arange(batch_size).to(self.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return data_loader

# Define the default logging directory
LOG_DIR = "/home/user/tilak/mbodied-agents/src/mbodied_agents/agents/motion/rt1/clip/logs"  # Change to your desired path

experiment_name = "clip_finetune"
logger = WandbLogger(project="clip", name=experiment_name)
# Step 2: Configure the Trainer with the W&B logger and checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=SAVE_PATH, 
    save_top_k=-1,  # Save all checkpoints
    save_weights_only=True, 
    every_n_epochs=1,  # Save checkpoint after every epoch
    filename='{epoch:02d}-{train_loss:.2f}',
)

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    devices=[0] if torch.cuda.is_available() else None,
    accelerator="gpu",
    limit_train_batches=STEPS_PER_EPOCH,  # Limit the number of training batches per epoch
    logger=logger,  # Add the W&B logger
    callbacks=[checkpoint_callback],  # Add the checkpoint callback
    default_root_dir=LOG_DIR,  # Set the default logging directory
)

# Create the model
model = CLIPFineTuner()

# Train the model
trainer.fit(model)
print("Fine-tuning complete and model saved.")

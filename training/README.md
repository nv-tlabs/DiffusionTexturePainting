# Train Diffusion Texture Painting

## Install dependencies

Before running the scripts, make sure to install the training dependencies:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Dataset
Download and unzip the DTD dataset [dtd-r1.0.1.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/dtd/) or use any other texture dataset. Our dataloader recursively searches for images in the "image_folder" with "png", "jpg", or "jpeg" file extensions.

If you need access to our pexels dataset for evaluation, please open an issue.

## Training with LoRA
**___Note: To monitor the training progress, we regularly generate sample images which are visualized on tensorboard (by default) or wandb (requires `pip install wandb`)___**


```bash
accelerate launch train_texture_inpaint_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
  --image_folder="dtd/images" \
  --resolution=256 \
  --train_batch_size=32 \
  --validation_epochs=1 \
  --cond_drop_prob=0.2 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="multi-scale-clip-patch-encoder-lora" \
  --report_to="tensorboard"
```

Resume from checkpoint by adding 
```bash
  --resume_from_checkpoint="checkpoint-1000"
```


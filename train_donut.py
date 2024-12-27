import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from huggingface_hub import HfFolder, login
import multiprocessing
import torch

# Log in to HuggingFace Hub
login(token="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")  # Replace with your access token

# Set Paths
# BASE_PATH = Path("/var/www/project/ai/")  # Change this as needed
BASE_PATH = Path("/content/test/")  # Change this as needed
IMAGE_PATH = BASE_PATH / "img"
ANNOTATIONS_PATH = BASE_PATH / "img/metadata.jsonl"

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Check for GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set PyTorch to use all CPU threads
torch.set_num_threads(num_cores)

# Load Donut Processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Add custom tokens if required
new_special_tokens = ["<s_custom>", "</s_custom>", "<sep/>", "<s>", "</s>"]
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

# Preprocessing function
def preprocess(sample):
    """Prepare the dataset for training."""
    text = sample["text"]
    doc = f"<s>{text}</s>"
    sample["text"] = doc

    # Ensure image is RGB
    sample["image"] = sample["image"].convert("RGB")
    return {"image": sample["image"], "text": sample["text"]}

# Load and preprocess dataset
dataset = load_dataset("imagefolder", data_dir=str(IMAGE_PATH), split="train")
processed_dataset = dataset.map(preprocess, num_proc=num_cores)  # Use all CPU cores

# Transform dataset into pixel values and token IDs
def transform_and_tokenize(sample):
    pixel_values = processor(
        sample["image"], random_padding=True, return_tensors="pt"
    ).pixel_values.squeeze(0)  # Ensure a single image tensor, not a batch

    input_ids = processor.tokenizer(
        sample["text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )["input_ids"].squeeze(0)  # Ensure single sequence tensor, not a batch

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens
    return {"pixel_values": pixel_values, "labels": labels}

# Define a custom data collator
class CustomDataCollator:
    def __call__(self, features):
        # Stack `pixel_values` into a batch
        pixel_values = torch.stack([
            torch.tensor(feature["pixel_values"]) if isinstance(feature["pixel_values"], list) else feature["pixel_values"]
            for feature in features
        ])

        # Stack `labels` into a batch, ensuring they are tensors
        labels = torch.stack([
            torch.tensor(feature["labels"]) if isinstance(feature["labels"], list) else feature["labels"]
            for feature in features
        ])

        # Return the batch as a dictionary
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# Apply transformations
processed_dataset = processed_dataset.map(
    transform_and_tokenize, remove_columns=["image", "text"], num_proc=num_cores  # Use all CPU cores
)

# Train-test split
split_dataset = processed_dataset.train_test_split(test_size=0.1)

# Load Pretrained Donut Model
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Adjust model configurations
if hasattr(processor.image_processor, "size"):
    image_size = processor.image_processor.size
else:
    image_size = (480, 640)  # Default or expected dimensions
model.config.encoder.image_size = image_size  # Use dimensions directly
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Move model to GPU if available
model.to(device)

# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="donut-custom",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size to 1
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="no",
    predict_with_generate=True,
    dataloader_num_workers=0,  # Use fewer workers for data loading
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    fp16=True,  # Enable mixed precision
    push_to_hub=False,  # Disable pushing to hub (optional)
    hub_model_id="donut-custom",
    hub_token=HfFolder.get_token(),  # You can also pass login token explicitly
)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove the `num_items_in_batch` argument before calling the model
        inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=CustomDataCollator(),
    tokenizer=processor,  # Replace with the correct processor if needed
)

# Train the Model
trainer.train()

# Save Model and Processor
model.save_pretrained("donut-custom")
processor.save_pretrained("donut-custom")

# Push to HuggingFace Hub
trainer.push_to_hub()

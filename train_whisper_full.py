import pandas as pd
import soundfile as sf
import torch

from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

print("Loading dataset...")

df = pd.read_csv("dataset.csv")

train_size = int(0.9 * len(df))

train_df = df[:train_size]
test_df = df[train_size:]

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Loading processor...")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny"
)

print("Loading model...")

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny"
)

# Set Assamese transcription task
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

def prepare_dataset(batch):

    audio, sample_rate = sf.read(batch["audio"])

    inputs = processor(
        audio,
        sampling_rate=sample_rate
    )

    batch["input_features"] = inputs.input_features[0]

    batch["labels"] = processor.tokenizer(
        batch["text"]
    ).input_ids

    return batch

print("Preparing datasets...")

train_dataset = train_dataset.map(prepare_dataset)
test_dataset = test_dataset.map(prepare_dataset)

train_dataset = train_dataset.remove_columns(
    ["audio", "text"]
)

test_dataset = test_dataset.remove_columns(
    ["audio", "text"]
)

def data_collator(features):

    input_features = torch.tensor(
        [f["input_features"] for f in features],
        dtype=torch.float32
    )

    labels_batch = processor.tokenizer.pad(
        [{"input_ids": f["labels"]} for f in features],
        return_tensors="pt"
    )

    labels = labels_batch["input_ids"]

    labels = labels.masked_fill(
        labels == processor.tokenizer.pad_token_id,
        -100
    )

    return {
        "input_features": input_features,
        "labels": labels
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-assamese",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

print("Starting training...")

trainer.train()

print("Saving model...")

model.save_pretrained(
    "./whisper-assamese"
)

processor.save_pretrained(
    "./whisper-assamese"
)

print("Training completed!")
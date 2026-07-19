import pandas as pd
import soundfile as sf
from transformers import WhisperProcessor

print("Loading processor...")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

df = pd.read_csv("dataset.csv")

sample = df.iloc[0]

audio, sample_rate = sf.read(sample["audio"])

input_features = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
).input_features

labels = processor.tokenizer(
    sample["text"],
    return_tensors="pt"
).input_ids

training_example = {
    "input_features": input_features,
    "labels": labels
}

print("Training example created!")
print("Input shape:", training_example["input_features"].shape)
print("Label shape:", training_example["labels"].shape)
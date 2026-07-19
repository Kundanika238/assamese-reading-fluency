import torch
import soundfile as sf

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

processor = WhisperProcessor.from_pretrained(
    "./whisper-assamese"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "./whisper-assamese"
)

audio, sr = sf.read("test.wav")

inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

with torch.no_grad():
    predicted_ids = model.generate(
        inputs.input_features
    )

text = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print("\nPrediction:")
print(text)
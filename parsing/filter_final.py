import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from datasets import Dataset

# set environment variables for debugging (optional)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#loading the summarization model with optimized settings
summarizer = pipeline(
    "summarization",
    model="IlyaGusev/mbart_ru_sum_gazeta",
    device=0 if device == "cuda" else -1,
    batch_size=1,  #before was 2
    torch_dtype=torch.float16 if device == "cuda" else None
)

df = pd.read_csv("data_GMKN.csv")

#splitting data into two groups
long_texts = df[df["text"].str.split().str.len() > 50].copy()
short_texts = df[df["text"].str.split().str.len() <= 50].copy()


dataset = Dataset.from_pandas(long_texts) #convert long texts into Hugging Face dataset for efficient batch processing

def summarize_batch(batch):
    texts = batch["text"]

    max_input_length = 1024
    texts = [t[:max_input_length] for t in texts]

    try:
        summaries = summarizer(texts, max_length=50, min_length=10, do_sample=False)
        batch["text"] = [s["summary_text"] for s in summaries]
    except RuntimeError as e:
        print(f"Error processing batch: {e}")
        batch["text"] = [""] * len(texts)

    return batch

batch_size = 2  #before was 4, gives error
long_texts = dataset.map(summarize_batch, batched=True, batch_size=batch_size, desc="Summarizing texts")

long_texts = long_texts.to_pandas()

print("Starting batch processing:")

#saving progress after each batch
for i in tqdm(range(0, len(long_texts), batch_size), desc="💾 Saving progress"):
    temp_df = pd.concat([long_texts.iloc[:i+batch_size], short_texts]).sort_index()
    temp_df.to_csv("resume_data_GMKN.csv", index=False)
    print(f"Saved progress: {i+batch_size}/{len(long_texts)} texts summarized.")

    torch.cuda.empty_cache()

    print("Progress saved! Continuing...\n")

#concatenate resumes
combined_df = pd.concat([long_texts, short_texts]).sort_index()
combined_df.to_csv("resume_data_GMKN.csv", index=False)

print("Processing complete! Data saved to 'resume_data_GMKN.csv'.")

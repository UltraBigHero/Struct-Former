import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Path to the local monologg emotion classification model(Our experiment chooses monologg to label dataset)
model_path = "./monologg"  # This directory contains config.json, pytorch_model.bin, etc.

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# List of GoEmotions fine-grained emotion labels
goemotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "embarrassment", "excitement",
    "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Mapping of fine labels to coarse categories
coarse_map = {
    "joy": ["joy", "amusement", "pride", "excitement", "relief", "optimism"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "embarrassment", "nervousness"],
    "surprise": ["surprise", "realization", "confusion"],
    "love": ["love", "caring", "admiration", "gratitude", "desire"],
    "neutral": ["neutral", "curiosity", "approval"]
}
fine2coarse = {fine: coarse for coarse, fines in coarse_map.items() for fine in fines}

# Load the structured lyrics from DALI dataset
def load_lyrics(path):
    return pd.read_excel(path)

# Predict the most probable fine and corresponding coarse label
def predict_hard_labels(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_idx = logits.argmax(dim=-1).item()
    fine = goemotion_labels[pred_idx] if pred_idx < len(goemotion_labels) else "neutral"
    coarse = fine2coarse.get(fine, "neutral")
    return fine, coarse

# Apply predictions to each line in the lyrics
def annotate(df):
    fine_labels = []
    coarse_labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Predicting"):
        text = str(row["line_text"]).strip()
        if not text or len(text) < 3:
            fine, coarse = "neutral", "neutral"
        else:
            fine, coarse = predict_hard_labels(text)
        fine_labels.append(fine)
        coarse_labels.append(coarse)
    df["fine_label"] = fine_labels
    df["coarse_label"] = coarse_labels
    return df[["song_id", "para_id", "line_id", "line_text", "fine_label", "coarse_label"]]

# Main pipeline
def main():
    df = load_lyrics("dali_lyrics_structured.xlsx")  # The input structured lyrics
    df_out = annotate(df)
    df_out.to_csv("labels.csv", index=False)
    print("✅ Finished: Saved output to labels.csv")

if __name__ == "__main__":
    main()

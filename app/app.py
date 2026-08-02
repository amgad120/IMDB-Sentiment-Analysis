import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "distilbert")
)

MODEL_PATH = os.path.abspath(MODEL_PATH)

LABELS = {
    0: "Negative 😞",
    1: "Positive 😊",
}

# ---------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_sentiment(text):
    if not text or not text.strip():
        return {"Please enter a review": 1.0}

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    return {
        LABELS[0]: float(probabilities[0]),
        LABELS[1]: float(probabilities[1]),
    }


# ---------------------------------------------------------
# Gradio application
# ---------------------------------------------------------
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=7,
        label="Movie Review",
        placeholder="Write a movie review here...",
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Sentiment Prediction",
    ),
    title=" IMDB Movie Review Sentiment Analyzer",
    description=(
        "A DistilBERT-based NLP application that classifies "
        "movie reviews as Positive or Negative."
    ),
    examples=[
        [
            "The movie was absolutely fantastic. "
            "The story was emotional and the acting was incredible."
        ],
        [
            "The acting was terrible and the dialogue felt unnatural. "
            "I couldn't wait for it to end."
        ],
        [
            "The movie started well, but the ending was disappointing "
            "and ruined the whole experience."
        ],
    ],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()

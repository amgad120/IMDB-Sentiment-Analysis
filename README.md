# 🎬 IMDB Sentiment Analysis

An end-to-end Natural Language Processing project for **binary sentiment classification of movie reviews** using traditional Machine Learning, Deep Learning, and Transformer-based models.

The project compares four approaches:

- Logistic Regression
- Linear SVM
- Bidirectional LSTM (BiLSTM)
- DistilBERT

The final DistilBERT model is also connected to a **Gradio web interface** for real-time sentiment prediction.

---

## 📌 Project Overview

The goal is to predict whether a movie review expresses a **Positive** or **Negative** sentiment.

### Example

```text
Input:
"The movie was amazing and I really enjoyed it."

Output:
Positive
```

This project covers the complete NLP workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Text Preprocessing
   ↓
Feature Representation / Tokenization
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
DistilBERT Fine-Tuning
   ↓
Model Saving
   ↓
Gradio Deployment
```

---

## 📊 Dataset

The project uses the **IMDB Movie Review Dataset** containing **50,000 labeled reviews**.

For the final DistilBERT experiment, the dataset was split into:

| Split | Samples |
|---|---:|
| Training | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |
| **Total** | **50,000** |

The two classes are:

```text
0 → Negative
1 → Positive
```

The dataset is balanced between the two sentiment classes.

---

## 🔎 Exploratory Data Analysis

Before model training, the dataset was inspected to understand:

- Dataset shape
- Missing values
- Duplicate rows
- Class distribution
- Review length distribution
- Example reviews

This helped verify the quality and structure of the dataset before modeling.

---

## 🧹 Text Processing

Different representations were used depending on the model.

### Traditional Machine Learning

For Logistic Regression and Linear SVM, reviews were converted into numerical features using **TF-IDF**.

### BiLSTM

Text was converted into token sequences and padded to a fixed sequence length before being passed through an embedding layer and BiLSTM architecture.

### DistilBERT

The original review text was tokenized using the **DistilBERT tokenizer**.

The final Transformer experiment used:

```text
Model: distilbert-base-uncased
Maximum sequence length: 256
Number of classes: 2
Learning rate: 2e-5
Batch size: 16
Epochs: 3
Weight decay: 0.01
```

---

## 🤖 Models Compared

### 1. Logistic Regression

A strong and efficient baseline for text classification using TF-IDF features.

**Test Accuracy: 89.94%**

### 2. Linear SVM

A linear Support Vector Machine trained on TF-IDF features.

**Test Accuracy: 90.14%**

### 3. BiLSTM

A Bidirectional LSTM neural network designed to capture sequential information from both directions of the text.

**Test Accuracy: 89.12%**

### 4. DistilBERT 🏆

A pretrained Transformer model fine-tuned for binary sentiment classification.

**Test Accuracy: ~91.88%**

---

## 📈 Model Comparison

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | **89.94%** |
| Linear SVM | **90.14%** |
| BiLSTM | **89.12%** |
| **DistilBERT** | **~91.88%** 🏆 |

DistilBERT achieved the best test performance among the evaluated models.

The improvement over the Linear SVM baseline was approximately **1.74 percentage points**.

---

## 🧪 DistilBERT Evaluation

The final test evaluation contained **5,000 previously unseen reviews**.

The classification report was approximately:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.92 | 0.92 | 0.92 |
| Positive | 0.92 | 0.92 | 0.92 |
| **Overall Accuracy** | | | **~0.92** |

The balanced performance across both classes indicates that the model was not strongly biased toward one sentiment class.

---

## 🧠 Training Behavior

The DistilBERT training results were:

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.2378 | 0.2315 | 91.48% |
| 2 | 0.1676 | 0.2475 | 91.38% |
| 3 | 0.1026 | 0.3137 | 91.88% |

The increasing validation loss while training loss continued to decrease suggests that the model started showing signs of **overfitting**. For this reason, simply increasing the number of epochs would not necessarily improve generalization.

---

## 🔬 New Review Testing

The final DistilBERT model was also tested on seven manually written reviews that were not part of the dataset.

The model correctly classified all seven examples according to their intended sentiment in this small qualitative test.

Example:

```text
"The acting was terrible and the dialogue felt unnatural.
I couldn't wait for it to end."

Prediction:
Negative 😞

Confidence:
99.83%
```

> Important: prediction confidence for an individual review should not be interpreted as the overall model accuracy. The overall performance is measured using the held-out test set.

---

## 🚀 Deployment

A lightweight **Gradio** interface was created to make the trained model interactive.

The application accepts a movie review and returns:

- Positive or Negative sentiment
- Model probability for each class

The application loads the saved DistilBERT model and tokenizer without retraining.

### Application flow

```text
User Review
     ↓
DistilBERT Tokenizer
     ↓
DistilBERT Classifier
     ↓
Softmax Probabilities
     ↓
Positive / Negative
     ↓
Confidence
```

---

## 📁 Project Structure

```text
IMDB-Sentiment-Analysis/
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── IMDB_Sentiment_Analysis.ipynb
│
├── models/
│   └── distilbert/
│       └── saved model files
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The trained model files are intentionally excluded from Git using `.gitignore` because Transformer checkpoints can be large. Store them separately or upload the model to a model hosting service.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd IMDB-Sentiment-Analysis
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 💾 Model Setup

The Gradio application expects the trained DistilBERT model at:

```text
models/distilbert/
```

The folder should contain the saved Hugging Face model and tokenizer files, such as:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

If the model is stored somewhere else, set the `MODEL_PATH` environment variable.

### Windows PowerShell

```powershell
$env:MODEL_PATH="C:\path\to\your\model"
python app/app.py
```

### Linux / macOS

```bash
export MODEL_PATH="/path/to/your/model"
python app/app.py
```

---

## ▶️ Run the Application

From the project root:

```bash
python app/app.py
```

Gradio will start the local web interface.

Open the URL displayed in the terminal.

---

## 🧪 Example Reviews

### Positive

```text
The movie was absolutely fantastic. The story was emotional and the acting was incredible.
```

### Negative

```text
The acting was terrible and the dialogue felt unnatural. I couldn't wait for it to end.
```

### Mixed

```text
The visuals were beautiful, but the story was confusing and the characters felt empty.
```

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Gradio
- Google Colab

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Natural Language Processing (NLP)
- Text preprocessing
- TF-IDF
- Logistic Regression
- Linear SVM
- Word/token embeddings
- Sequence modeling
- Bidirectional LSTM
- Transformers
- Transfer Learning
- Fine-Tuning
- Tokenization
- Model evaluation
- Classification reports
- Model persistence
- Inference
- Interactive AI deployment

---

## 🔮 Future Improvements

Possible next steps include:

- Hyperparameter optimization
- Better handling of mixed/neutral sentiment
- Data augmentation
- Cross-validation for traditional ML models
- More extensive error analysis
- Calibration of prediction probabilities
- Deploying the application to a cloud platform
- Publishing the trained model to Hugging Face Hub
- Adding an API layer using FastAPI

---

## 👤 Author

**Amgad Essam**

AI & Data-focused student interested in Machine Learning, Deep Learning, NLP, and practical AI applications.

---

## ⭐ Project Highlights

```text
50,000 Reviews
        ↓
4 Different Models
        ↓
DistilBERT ≈ 91.88% Test Accuracy
        ↓
Saved Model
        ↓
Interactive Gradio Application
```

If you find the project useful, feel free to ⭐ the repository.

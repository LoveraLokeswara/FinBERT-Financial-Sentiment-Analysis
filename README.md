# FinBERT Financial Sentiment Analysis

A comprehensive financial sentiment analysis project comparing multiple state-of-the-art NLP models for classifying financial text into positive, negative, or neutral sentiment categories.

## 📋 Overview

This project implements and compares various transformer-based models and deep learning architectures for financial sentiment analysis. It includes both pre-trained and fine-tuned versions of popular models, along with an interactive Gradio-based chatbot interface for real-time sentiment prediction.

## 🎯 Key Features

- **Multiple Model Implementations**: FinBERT, BERT, RoBERTa, and ULMFiT
- **Fine-tuning Capabilities**: Custom fine-tuning on Financial PhraseBank dataset
- **Interactive Dashboard**: Gradio-based web interface for sentiment prediction
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Visualization**: Confusion matrices and comparative performance plots

## 📊 Models Tested

| Model | Type | Accuracy | Notes |
|-------|------|----------|-------|
| **FinBERT (Fine-tuned)** | Fine-tuned | 97.35% | Best performing model |
| **FinBERT (Pre-trained)** | Pre-trained | 91.70% | Domain-specific model |
| **ULMFiT** | Fine-tuned | 79.87% | Language model approach |
| **BERT (Fine-tuned)** | Fine-tuned | 76.82% | General transformer |
| **RoBERTa** | Pre-trained | 70.54% | Twitter-sentiment based |
| **BERT (Pre-trained)** | Pre-trained | 58.96% | General purpose |

## 📂 Project Structure

```
FinBERT-Financial-Sentiment-Analysis/
│
├── FinBert_Final_Project_CS679.ipynb       # Main project notebook
├── FinBert_Final_Project_CS679.pdf         # Project report
├── FinBert_Financial_Sentiment_Analysis.ipynb  # Additional analysis
├── FinBert_Financial_Sentiment_Analysis.pdf
├── Fin_Chatbot.ipynb                       # Gradio chatbot interface
├── Fin_Chatbot.pdf
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- GPU recommended for training (CPU works but slower)
- 8GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinBERT-Financial-Sentiment-Analysis.git
cd FinBERT-Financial-Sentiment-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 💻 Usage

### Training Models

Open `FinBert_Final.ipynb` to:
- Load and preprocess the Financial PhraseBank dataset
- Train/fine-tune models
- Evaluate performance with metrics and visualizations
- Save trained models

### Using the Chatbot Interface

Run `Fin_Chatbot.ipynb` to launch an interactive Gradio dashboard:

```python
# The interface will be available at http://127.0.0.1:7860
# Enter financial text to get sentiment predictions with probability scores
```

**Note**: You'll need a trained model saved locally. Update the `model_path` variable in the notebook to point to your model directory.

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "path/to/your/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict sentiment
text = "The stock price is rising today and showing strong momentum"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

labels = ["positive", "negative", "neutral"]
prediction = labels[probs.argmax()]
print(f"Sentiment: {prediction}")
```

## 📈 Datasets

### Financial PhraseBank
- **Source**: `takala/financial_phrasebank` (Hugging Face)
- **File**: `Sentences_AllAgree.txt`
- **Description**: Sentences from financial news categorized by sentiment with 100% annotator agreement

### Financial Sentiment Analysis
- **Source**: `mltrev23/financial-sentiment-analysis`
- **File**: `Fianancial_Sentiment_Analysis.csv`
- **Description**: Larger dataset of financial sentences with sentiment labels

## 🔧 Model Details

### FinBERT
- **Base Model**: `yiyanghkust/finbert-tone`
- **Fine-tuning**: 3 epochs, batch size 8
- **Tokenization**: Max length 128 tokens
- **Best Results**: 97.35% accuracy on test set

### BERT
- **Base Model**: `bert-base-uncased`
- **Fine-tuning**: 5 epochs, batch size 8
- **Architecture**: Standard BERT with classification head

### RoBERTa
- **Base Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Usage**: Pre-trained without additional fine-tuning

### ULMFiT
- **Architecture**: AWD-LSTM
- **Framework**: FastAI
- **Training**: 1 epoch fine-tuning with drop_mult=0.5

## 📊 Evaluation Metrics

The project evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **UI**: Gradio
- **Additional**: FastAI (for ULMFiT)

## 📝 Key Findings

1. **Domain-specific models excel**: FinBERT significantly outperforms general models due to pre-training on financial text
2. **Fine-tuning is crucial**: Fine-tuned versions consistently outperform pre-trained counterparts
3. **Class imbalance matters**: Models perform better on majority classes (positive/neutral)
4. **Context understanding**: Longer context windows improve sentiment classification accuracy


# 🤖 Customer Service Chatbot
### NLP-Based Intent Classification System | Codec Technologies AI Internship

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=for-the-badge&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**Built by [Ankita Ghosh](https://linkedin.com/in/ank1412) · AI Intern @ Codec Technologies**

[▶️ Open in Colab](https://colab.research.google.com) · [📊 Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) · [💼 LinkedIn](https://linkedin.com/in/ank1412)

</div>

---

## 📌 Project Overview

A production-ready **NLP-based Customer Service Chatbot** that understands customer intent and responds intelligently. The system uses **TF-IDF vectorization** and **Logistic Regression** to classify 27 unique customer support intent categories from real-world data, automatically downloaded from HuggingFace.

```
User Query → Preprocessing → TF-IDF → Intent Classifier → Response Generator → Answer
```

---

## 🎯 Key Features

| Feature | Description |
|--------|-------------|
| 🔄 **Auto Dataset Download** | Pulls 26,872 real customer queries from HuggingFace |
| 🧹 **NLP Preprocessing** | Tokenization, lemmatization, stopword removal |
| 📐 **TF-IDF Vectorization** | 5000 features with unigram + bigram support |
| 🤖 **Intent Classifier** | Logistic Regression with 27-class prediction |
| 📊 **Confidence Scoring** | Every prediction includes a confidence % |
| 💬 **Response Templates** | Human-like, varied responses per intent |
| 💾 **Model Persistence** | Save & reload with joblib |
| 📓 **Colab-Ready** | Fully interactive notebook with chat interface |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CUSTOMER SERVICE CHATBOT              │
├─────────────┬────────────────┬──────────────────────────┤
│  Input Layer│  NLP Pipeline  │    Response Layer        │
│             │                │                          │
│  User Query │→ Lowercase     │  Intent Mapper           │
│             │→ Remove Punct  │  Response Templates      │
│             │→ Tokenize      │  Confidence Filter       │
│             │→ Remove stops  │  Random Variation        │
│             │→ Lemmatize     │  Fallback Handler        │
│             │→ TF-IDF Vec    │                          │
│             │→ LR Classifier │                          │
└─────────────┴────────────────┴──────────────────────────┘
```

---

## 📦 Dataset

**Source:** [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

| Attribute | Value |
|-----------|-------|
| Total Records | 26,872 |
| Unique Intents | 27 |
| Language | English |
| Source | HuggingFace Hub |
| Auto-Download | ✅ Yes (no manual steps) |

**Intent Categories include:** `cancel_order`, `get_refund`, `track_order`, `payment_issue`, `contact_customer_service`, `delivery_options`, `create_account`, `recover_password`, and 19 more.

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

```
1. Open the notebook: Customer_Service_Chatbot.ipynb
2. Click Runtime → Run All
3. Dataset downloads automatically ✅
4. Scroll to Step 10 for the interactive chat!
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/AG141293/customer-service-chatbot-nlp.git
cd customer-service-chatbot-nlp

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Customer_Service_Chatbot.ipynb
```

---

## 📁 Project Structure

```
customer-service-chatbot-nlp/
│
├── 📓 Customer_Service_Chatbot.ipynb   ← Main notebook (Colab-ready)
├── 📄 README.md                         ← You're here
├── 📋 requirements.txt                  ← Dependencies
├── 📊 intent_distribution.png           ← EDA visualization
├── 📊 model_comparison.png              ← Model accuracy chart
└── 📁 saved_model/
    ├── intent_classifier.pkl            ← Trained LR model
    ├── tfidf_vectorizer.pkl             ← TF-IDF vectorizer
    └── label_encoder.pkl               ← Label encoder
```

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| ✅ Logistic Regression | **~93%+** |
| Naive Bayes (Baseline) | ~85% |

---

## 🔧 Tech Stack

```python
# Core NLP
nltk              # Tokenization, lemmatization, stopwords
scikit-learn      # TF-IDF, Logistic Regression, evaluation metrics

# Data
datasets          # HuggingFace auto-download
pandas            # Data manipulation
numpy             # Numerical operations

# Visualization
matplotlib        # Plots and charts
seaborn           # Statistical visualizations

# Model Persistence
joblib            # Save/load model artifacts
```

---

## 💬 Sample Interaction

```
🤖 Bot: Hello! Welcome to Customer Support. How can I help you today?

👤 You: I want to cancel my order
🤖 Bot: I understand you'd like to cancel your order. Orders can be cancelled within 24
        hours of placement. Please share your order ID and I'll process this right away.
        [Intent: cancel_order | Confidence: 97.3%]

👤 You: Where is my package?
🤖 Bot: I can track your order! Please share your order ID or tracking number and
        I'll give you a real-time update.
        [Intent: track_order | Confidence: 95.1%]

👤 You: How do I reset my password?
🤖 Bot: Password recovery is easy! Click 'Forgot Password' on our login page and
        follow the instructions sent to your registered email.
        [Intent: recover_password | Confidence: 98.6%]
```

---

## 🔮 Future Enhancements

- [ ] BERT/DistilBERT for transformer-based intent classification
- [ ] LangChain + RAG for knowledge-base-grounded responses
- [ ] Streamlit web app deployment
- [ ] SpaCy NER for entity extraction (order IDs, emails)
- [ ] Multi-turn conversation memory with LangGraph
- [ ] Multilingual support for Indian languages

---

## 👩‍💻 About the Author

**Ankita Ghosh** — GenAI & ML Engineer | RAG Architect | NLP Specialist

- 🎓 M.Sc. AI & ML — Woolf University (CGPA: 9.3)
- 🏢 AI Intern @ Codec Technologies (2026)
- 🏢 ML Intern @ Tech Mahindra Makers Lab — Project INDUS (India's LLM)
- 🏆 Top Learner @ Scaler | VLM Bootcamp 100% | Claude 101 (Anthropic)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/ank1412)
[![GitHub](https://img.shields.io/badge/GitHub-50+_repos-black?logo=github)](https://github.com/AG141293)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface)](https://huggingface.co/AnkGhosh)

---



---

<div align="center">
⭐ Star this repo if you found it helpful! · Built with ❤️ for Codec Technologies AI Internship
</div>


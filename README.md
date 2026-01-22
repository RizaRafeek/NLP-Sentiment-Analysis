# 🎭 Sentiment Analysis Engine: Neural Text Classification
> **Architecting a Deep Learning pipeline for binary sentiment classification of unstructured movie reviews.**

## 📌 Project Overview
Text data is inherently high-dimensional and sparse. This project implements an **Artificial Neural Network (ANN)** that utilizes a learned **Embedding Space** to map words into dense vectors, enabling the model to understand semantic relationships and classify sentiment with high generalization.

## 🛠️ Technical Implementation
* **Text Engineering:** * Implemented **Tokenization** and **Integer Encoding** for a 10,000-word vocabulary.
    * Utilized **Post-Padding (`pad_sequences`)** to ensure uniform input dimensions ($n=250$) across variable-length reviews.
* **Architecture:** * **Embedding Layer:** Maps 10,000 discrete tokens into a 16-dimensional dense vector space.
    * **Global Average Pooling:** Reduced the temporal dimension to prevent the model from becoming sensitive to specific word positions, focusing instead on global sentiment.
* **Regularization:** * Applied an aggressive **0.7 Dropout** rate to the bottleneck layer.
    * Integrated **Early Stopping** (patience=2) to prevent the "over-memorization" of training noise.

## 📊 Performance Analysis
| Metric | Result |
| :--- | :--- |
| **Vocabulary Size** | 10,000 words |
| **Embedding Dim** | 16 |
| **Optimization** | Adam |
| **Final State** | Balanced (Generalization-Focused) |



## 🚀 Key Engineering Insights
1. **Dense Embeddings vs. One-Hot:** Unlike sparse One-Hot encoding, the **Embedding Layer** allows the model to learn that "excellent" and "great" are mathematically similar, significantly improving performance on small datasets.
2. **Battling Overfitting:** Movie reviews often contain "noisy" words. The combination of **Global Average Pooling** and **0.7 Dropout** was essential to force the model to look at the overall review context rather than specific "trigger words."
3. **Architecture Choice:** For this specific task, a Global Average Pooling ANN was chosen over an RNN/LSTM to minimize computational cost while maintaining competitive accuracy for short-form text.

## 💻 Deployment & Reproducibility
* **Environment:** Python 3.10+, TensorFlow/Keras
* **Inference:** Load `sentiment_model.h5` and use the `word_index.json` to pre-process raw strings before prediction.

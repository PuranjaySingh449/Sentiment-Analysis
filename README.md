# Twitter Sentiment Classification: LSTM + TF‑IDF vs BERT

End‑to‑end comparison of a classic **LSTM + TF‑IDF** text classifier and a fine‑tuned **BERT Transformer** on a **4‑class Twitter sentiment dataset**: Irrelevant, Negative, Neutral, Positive.

## 📊 Results

On the held‑out Twitter validation set:

| Model         | Accuracy | F1‑Score |
|--------------|----------|----------|
| LSTM + TF‑IDF| 0.9520   | 0.9520   |
| **BERT**     | **0.9640** | **0.9640** |

BERT achieves slightly higher accuracy/F1 and fewer misclassifications, while LSTM + TF‑IDF remains a strong and lighter baseline.

## 🧠 Models

- **LSTM + TF‑IDF**
  - TF‑IDF vectorizer converts tweets to sparse vectors.
  - Single LSTM layer consumes the vector (reshaped to sequence) and outputs class probabilities.
  - Saved as `best_lstm_model.h5` with `tfidf_vectorizer.pkl` and `label_encoder.pkl`.

- **BERT**
  - Fine‑tuned `AutoModelForSequenceClassification` from Hugging Face.
  - Uses subword tokenization, attention and contextual embeddings.
  - Stored in a directory like `bert_sentiment_model/` containing config, tokenizer, and weights.

Both models predict the same 4 sentiment labels encoded by the shared `LabelEncoder`.

## 🗂️ Project Structure

```text
twitter-sentiment/
├── model_comparison.py        # LSTM vs BERT evaluation + plots
├── best_lstm_model.h5         # Trained LSTM model
├── tfidf_vectorizer.pkl       # TF‑IDF vectorizer for LSTM
├── label_encoder.pkl          # Class encoder (4 labels)
├── bert_sentiment_model/      # Fine‑tuned BERT checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── twitter_validation.csv     # Validation set: id, platform, label, text
└── README.md
```

Recommended `.gitignore` (if you put this on GitHub):

```gitignore
*.h5
*.pkl
bert_sentiment_model/
*.csv
__pycache__/
*.pyc
```

## 🚀 How to Run

1. Install dependencies (example):

```bash
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib seaborn
```

2. Place the following files in the project folder:

- `best_lstm_model.h5`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `bert_sentiment_model/` (folder)
- `twitter_validation.csv`

3. Run the comparison script:

```bash
python model_comparison.py
```

This will:

- Load the validation tweets.
- Run predictions with:
  - `predict_lstm()` (TF‑IDF → LSTM)
  - `predict_bert()` (tokenizer → BERT)
- Print the metrics table above.
- Show:
  - Bar chart of **Accuracy & F1‑Score**
  - Confusion matrices for both models
  - Bar chart of misclassification counts
  - A text list of the top disagreements between LSTM and BERT.

## 🔍 Interpretation

- **LSTM + TF‑IDF** already reaches **95.2%** accuracy → great strong baseline.
- **BERT** pushes performance to **96.4%** and reduces errors in all four sentiment classes.
- Confusion matrices show that both models are best on Neutral/Positive, with BERT slightly better at avoiding cross‑class confusion (e.g. Negative vs Neutral).


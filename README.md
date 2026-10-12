# Amazon Reviews Sentiment Analysis

An NLP project focused on classifying Amazon customer reviews as positive or negative using traditional machine learning and deep learning techniques.

The project compares a **TF-IDF + Logistic Regression** baseline against a fine-tuned **BERT transformer**, evaluating their performance and exploring patterns in customer feedback.

## Project Overview

The objective is to build and evaluate sentiment classification models capable of identifying customer opinions from product reviews.

The project covers:

- Data loading, cleaning, and preprocessing
- Feature extraction using TF-IDF
- Sentiment classification using Logistic Regression
- Fine-tuning a pretrained BERT model
- Model evaluation and performance comparison
- Text analysis using bigrams and word clouds

## Dataset

The project uses the [Amazon Reviews Polarity Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews), available on Kaggle.

The original dataset contains **4 million Amazon product reviews**:

| Split | Reviews |
|---|---:|
| Training | 3,600,000 |
| Testing | 400,000 |

Each review contains a sentiment label, review title, and review text.

The dataset has two sentiment classes:

- **Negative:** Label 1
- **Positive:** Label 2

For this project, 200,000 reviews were randomly sampled from the original training dataset. The labels were converted to 0 (negative) and 1 (positive).

The sampled data was divided into:

- 160,000 training reviews (80%)
- 40,000 validation reviews (20%)

Stratified sampling was used to preserve the class distribution.

## Methodology

### 1. Data Preprocessing

The review text was processed through several cleaning steps:

- Decoding HTML entities
- Removing emojis and URLs
- Removing punctuation
- Normalizing whitespace
- Combining review titles and review text
- Encoding sentiment labels for binary classification

### 2. TF-IDF + Logistic Regression

A Logistic Regression model was implemented as the baseline classifier.

TF-IDF was used to convert review text into numerical features, incorporating both unigrams and bigrams.

**Configuration:**

| Parameter | Value |
|---|---|
| Maximum TF-IDF features | 20,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 3 |
| Maximum Logistic Regression iterations | 1,000 |

### 3. BERT Fine-Tuning

The pretrained `bert-base-uncased` transformer was fine-tuned for binary sentiment classification using PyTorch and Hugging Face Transformers.

**Training configuration:**

| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Epochs | 2 |
| Batch size | 16 |
| Maximum sequence length | 128 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Model selection | Best validation accuracy |

Dynamic padding was applied during tokenization, and GPU acceleration was used when available.

## Model Evaluation

Both models were evaluated using:

- Accuracy
- F1-score
- Classification reports
- Confusion matrices
- ROC curves and ROC-AUC

### Results

The following results were recorded on the 40,000-review validation set:

| Metric | Logistic Regression | BERT |
|---|---:|---:|
| Accuracy | 91.74% | 95.74% |
| F1-score | 91.76% | 95.74% |
| ROC-AUC | 91.74%* | 99.07% |

**BERT achieved an improvement of approximately 4 percentage points in accuracy** over Logistic Regression.

*Note: The baseline ROC-AUC was calculated using predicted class labels, while BERT's ROC-AUC used class probabilities. These scores are not directly comparable.*

These results represent validation performance rather than performance on the original dataset's separate test split.

## Text Analysis

Additional exploratory analysis was performed to identify common expressions associated with positive and negative reviews.

This included:

- Extracting frequent bigrams for each sentiment class
- Generating word clouds for positive and negative reviews
- Comparing common language patterns across customer opinions

These visualizations provide insight into frequently occurring terms and phrases in customer feedback.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- WordCloud
- Jupyter Notebook

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn torch transformers datasets accelerate emoji wordcloud tqdm jupyter
```

### 3. Download the dataset

Download the [Amazon Reviews Polarity Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) from Kaggle.

Extract the dataset into the project directory so that the expected file structure is:

```text
project/
├── NLP_Amazon_Reviews_Analysis.ipynb
└── amazon_review_polarity_csv/
    ├── train.csv
    └── test.csv
```

The notebook uses `train.csv`. The original test file is not used in the recorded evaluation.

### 4. Run the notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `NLP_Amazon_Reviews_Analysis.ipynb` and execute the cells in sequence.

The notebook will:

1. Load and preprocess the dataset.
2. Train the Logistic Regression baseline.
3. Fine-tune BERT.
4. Evaluate and compare both models.
5. Generate text visualizations.
6. Save the fine-tuned BERT model.

The trained BERT model and tokenizer are saved in the `bert_sentiment/` directory.

**Execution note:** The notebook includes a BERT inference cell that loads the model from `bert_sentiment/`. When running from scratch, ensure the model and tokenizer have been saved before executing that cell. The active sampling code uses 200,000 reviews, despite some outdated configuration comments.

## Conclusion

This project demonstrates an end-to-end NLP sentiment analysis workflow, combining traditional machine learning and transformer-based deep learning.

The experimental results show that fine-tuned BERT outperformed the TF-IDF and Logistic Regression baseline on the validation dataset.

The project also demonstrates the application of text preprocessing, transfer learning, model evaluation, and exploratory text analysis to a large-scale customer review dataset.

## Dataset Reference

Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.*

Dataset: [Amazon Reviews Polarity — Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

# 🎓 Education Policy Sentiment Analysis

This project focuses on **sentiment analysis of educational policy-related text data** to understand public opinion on educational reforms such as *NEP 2020 (National Education Policy)*.  
Multiple machine learning models classify text into **positive**, **negative**, or **neutral** sentiments and their performance is compared using standard evaluation metrics.

NLP techniques help interpret feedback on policies and government decisions from online discussions, news headlines, and articles.

---

## 📊 Problem Statement

Education policies directly impact millions of students and teachers. Identifying how the public perceives such policies can guide better decision-making and reforms.  
The challenge: automatically analyze user opinions/statements related to educational policies, categorizing them into sentiment classes.

---

## 📂 Dataset Details

### 1️⃣ Training Dataset — NEP 2020 English Tweets

English tweets related to India’s NEP 2020, labeled with their public sentiment.

| Sentiment Category | Description | Number of Tweets |
|--------------------|------------------------------------------------|------------------|
| Positive           | Tweets expressing favorable views about NEP 2020 | 3200             |
| Negative           | Tweets showing disagreement or criticism         | 2950             |
| Neutral            | Tweets with balanced or factual statements       | 3100             |

<img src="sentiment-chart-small.png" width="420"/>

**Key Preprocessing Steps**
- Text cleaning (punctuation, URLs, emojis, and stopword removal)  
- Tokenization and normalization  
- TF-IDF feature extraction for text representation  

📘 **Dataset Source:** `NEP_2020_english_tweet.csv`

---

### 2️⃣ Testing Dataset — NEP 2020 Test Tweets

Unseen tweets about NEP 2020 for evaluating trained models.

| Sentiment Category | Description | Sample Size |
|--------------------|----------------------------------|------------|
| Positive           | Supportive tweets about NEP 2020 | 800        |
| Negative           | Tweets expressing opposition     | 750        |
| Neutral            | Informative or balanced statements| 850        |

<img src="test-chart-small.png" width="420"/>

This cross-dataset evaluation ensures classifier reliability.

📘 **Dataset Source:** `test.csv`

---

## ⚙️ Methodology

1️⃣ **Data Preprocessing**
- Clean and normalize textual data  
- Apply TF-IDF vectorization  

2️⃣ **Model Training**
- Algorithms: Logistic Regression, SVM, Naive Bayes  
- All models use same TF-IDF features

3️⃣ **Evaluation**
- Accuracy, Precision, Recall, F1-Score  
- Compared models on identical test data

4️⃣ **Deployment**
- Interactive **Streamlit web app** for input, model selection, and instant prediction

---

## 🧠 Models Used

| Model                 | Description                                                      | Accuracy |
|-----------------------|------------------------------------------------------------------|----------|
| Logistic Regression   | Linear baseline, interpretable and efficient                    | 0.87     |
| SVM                   | Maximizes margin, robust for small text datasets                | 0.83     |
| Naive Bayes           | Probabilistic, assumes feature independence, fast for text      | 0.79     |

---

## Model Training and Evaluation

Models trained with **TF-IDF features**, validated using 80%-20% train-test split.

### NLP Preprocessing Pipeline

| Step                | Description                                                        | Libraries Used                                |
|---------------------|--------------------------------------------------------------------|-----------------------------------------------|
| Data Cleaning       | Remove punctuation, URLs, digits, emojis, lowercase conversion     | `re`, `string`                               |
| Tokenization        | Split tweets into tokens                                           | `nltk.word_tokenize`                         |
| Stopword Removal    | Eliminate common non-informative words                            | `nltk.corpus.stopwords`                      |
| Lemmatization       | Basic form conversion (e.g. “running” → “run”)                    | `WordNetLemmatizer`                          |
| TF-IDF Vectorization| Transform cleaned text into feature vectors                       | `TfidfVectorizer(max_features=5000, stop_words='english')` |

---

### Training Results and Comparison

| Model                | Accuracy | Precision | Recall | F1-Score | Notes                                                                          |
|----------------------|----------|-----------|--------|----------|--------------------------------------------------------------------------------|
| Logistic Regression  | 91.42%   | 0.913     | 0.911  | 0.912    | Strong linear baseline, interpretable                                          |
| Naive Bayes          | 90.87%   | 0.909     | 0.906  | 0.907    | Performs well with smaller text datasets                                       |
| Random Forest        | 92.35%   | 0.924     | 0.922  | 0.923    | Slightly better generalization on unseen tweets                                |
| SVM (Best)           | 93.12%   | 0.931     | 0.930  | 0.930    | Best model, balanced precision-recall                                          |

<img src="training-comparison-small.png" width="420"/>

SVM outperformed others in accuracy and stability.

---

### Testing and Cross-Dataset Validation (NEP 2020 Dataset)

Evaluation on the NEP 2020 test set (unseen tweets).

| Metric    | Logistic Regression | Naive Bayes | SVM    |
|-----------|--------------------|-------------|--------|
| Accuracy  | 0.9512             | 0.9438      | 0.9481 |
| Precision | 0.9560             | 0.9492      | 0.9514 |
| Recall    | 0.9487             | 0.9376      | 0.9423 |
| F1-Score  | 0.9523             | 0.9433      | 0.9468 |

<img src="test-comparison-small.png" width="420"/>

**Observation:**  
Logistic Regression shows best overall generalization.  
Naive Bayes & SVM maintain reliable accuracy (~94%).

---

## 📈 Results Summary

- Logistic Regression: 87% accuracy  
- SVM: 83% accuracy  
- Naive Bayes: 79% accuracy  

Reliable NLP-based sentiment classification for education policy analysis.

---

## 🌐 Streamlit Web Application

Interactive **Streamlit** web app using trained **TF-IDF** & **Logistic Regression** model.

🔗 **Live App:** [https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)

---

## 🧭 How the App Works

### 📝 Input

Users can enter one or more tweets about NEP 2020 (separated by new lines).

**Example Input:**
```
NEP 2020 will revolutionize the education system in India.
The implementation process needs more clarity and transparency.
I’m still unsure about how NEP 2020 affects higher education.
```

---

### ⚙️ Processing Pipeline

- Cleaning (URLs, emojis, stopwords)
- TF-IDF vectorization
- Sentiment classification model prediction
- Color-coded result visualization

---

### 📋 Example Output

| Tweet                                                       | Predicted Sentiment |
|-------------------------------------------------------------|---------------------|
| NEP 2020 will revolutionize the education system in India.   | Positive            |
| The implementation process needs more clarity and transparency.| Negative          |
| I’m still unsure about how NEP 2020 affects higher education.| Neutral             |

---

## 🖥️ How to Run the Streamlit App Locally

### Step 1️⃣ – Clone the Repository  
```
git clone https://github.com/arya-rehpade/Education-Policy-Sentiment.git
cd Education-Policy-Sentiment
```

### Step 2️⃣ – Install Dependencies  
```
pip install -r requirements.txt
```

### Step 3️⃣ – Launch the App  
```
streamlit run app.py
```

---

## 👩‍💻 Author

**Arya Rehpade**  
🎓 Machine Learning & Data Science Enthusiast  
📍 India

**GitHub:** [arya-rehpade](https://github.com/arya-rehpade)  
**Live App:** [Streamlit App](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)
**Repository:** Education-Policy-Sentiment

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute for educational/research purposes, provided you give proper credit.

---

```

**Instructions:**  
- Make sure your image files (`sentiment-chart-small.png`, `test-chart-small.png`, `training-comparison-small.png`, `test-comparison-small.png`) are present in the repo and use the correct names; adjust the `<img src="">` filename or path if needed.  
- All images now display smaller for easier embedding in GitHub.

**Just copy and paste the code above in your README.md!** Let me know if you want a different width or to add/remove any section.

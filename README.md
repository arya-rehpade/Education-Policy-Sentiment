# 🎓 Education Policy Sentiment Analysis  

This project focuses on **sentiment analysis of educational policy-related text data** to understand public opinion on educational reforms such as *NEP 2020 (National Education Policy)*.  
The main objective is to build multiple machine learning models that classify text into **positive**, **negative**, or **neutral** sentiments, and then compare their performance using standard evaluation metrics.  

The project demonstrates how **Natural Language Processing (NLP)** techniques can help interpret feedback on policies and government decisions from online discussions, news headlines, and articles.  

---

## 📊 Problem Statement  

Education policies directly impact millions of students and teachers. Identifying how the public perceives such policies can guide better decision-making and reforms.  
The challenge was to automatically analyze user opinions or statements related to educational policies and categorize them into sentiment classes.  

---

## 📂 Dataset Details  

### 1️⃣ Training Dataset — NEP 2020 English Tweets  

This dataset contains English tweets related to India’s National Education Policy (NEP 2020).  
Each tweet is labeled with a sentiment category indicating the public opinion toward the policy.  

| Sentiment Category | Description                                      | Number of Tweets |
| ------------------ | ------------------------------------------------ | ---------------- |
| Positive           | Tweets expressing favorable views about NEP 2020 | 3200             |
| Negative           | Tweets showing disagreement or criticism         | 2950             |
| Neutral            | Tweets with balanced or factual statements       | 3100             |

![Training Dataset](https://github.com/user-attachments/assets/6f579d89-e333-4913-a181-0be54de5bf1c)

**Key Preprocessing Steps**
- Text cleaning (punctuation, URLs, emojis, and stopword removal)  
- Tokenization and normalization  
- TF-IDF feature extraction for text representation  

📘 **Dataset Source:** `NEP_2020_english_tweet.csv`

---

### 2️⃣ Testing Dataset — NEP 2020 Test Tweets  

This testing dataset is a smaller subset containing unseen tweets about NEP 2020.  
It helps evaluate how well trained models generalize to new data.  

| Sentiment Category | Description                        | Sample Size |
| ------------------ | ---------------------------------- | ----------- |
| Positive           | Supportive tweets about NEP 2020   | 800         |
| Negative           | Tweets expressing opposition       | 750         |
| Neutral            | Informative or balanced statements | 850         |

![Testing Dataset](https://github.com/user-attachments/assets/69550d52-0b1f-4195-95f4-ee66f90f1002)

📘 **Dataset Source:** `test.csv`

---

## ⚙️ Methodology  

1️⃣ **Data Preprocessing**
- Clean and normalize textual data  
- Apply TF-IDF vectorization to transform text into numerical format  

2️⃣ **Model Training**
- Implemented three machine learning algorithms:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Naive Bayes  
- Models were trained on the same TF-IDF features for fair comparison  

3️⃣ **Evaluation**
- Used Accuracy, Precision, Recall, and F1-Score metrics  
- Compared all three models on identical test data  

4️⃣ **Deployment**
- Built an interactive **Streamlit web app** that allows users to input custom text, select models, and instantly view sentiment predictions  

---

## 🧠 Models Used  

| Model | Description | Accuracy |
|--------|--------------|-----------|
| Logistic Regression | Performs well for linearly separable data; interpretable and efficient for text classification. | 0.87 |
| SVM (Support Vector Machine) | Maximizes class separation margin; robust for small text datasets. | 0.83 |
| Naive Bayes | Probabilistic model assuming feature independence; fast and effective for text. | 0.79 |

---

## 🧩 Model Training and Evaluation  

All models were trained on **TF-IDF features** and validated using an **80%-20% train-test split**.  

### NLP Preprocessing Pipeline  

| Step | Description | Libraries Used |
| :---- | :----------- | :-------------- |
| Data Cleaning | Removed punctuation, URLs, digits, emojis, and converted text to lowercase | `re`, `string` |
| Tokenization | Split tweets into tokens (words) | `nltk.word_tokenize` |
| Stopword Removal | Eliminated common non-informative words like “the”, “is”, “and” | `nltk.corpus.stopwords` |
| Lemmatization | Converted words to their base form (e.g., “running” → “run”) | `WordNetLemmatizer` |
| TF-IDF Vectorization | Transformed cleaned text into vectors | `TfidfVectorizer(max_features=5000, stop_words='english')` |

---

### Training Results and Comparison  

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
| :----- | :-------: | :--------: | :------: | :--------: | :------ |
| Logistic Regression | 91.42% | 0.913 | 0.911 | 0.912 | Strong linear baseline |
| Naive Bayes | 90.87% | 0.909 | 0.906 | 0.907 | Performs well with smaller text data |
| Random Forest | 92.35% | 0.924 | 0.922 | 0.923 | Slightly better generalization |
| **SVM (Best)** | **93.12%** | **0.931** | **0.930** | **0.930** | Best performing model overall |

![Model Comparison](https://github.com/user-attachments/assets/00e91471-532e-467e-9415-8de0e0aebe4d)

---

## 🧮 Testing and Cross-Dataset Validation  

Evaluation was performed on unseen **NEP 2020 Test Tweets** to ensure model reliability.  

| Metric | Logistic Regression | Naive Bayes | SVM |
| :------ | :----------------: | :----------: | :---: |
| Accuracy | 0.9512 | 0.9438 | 0.9481 |
| Precision | 0.9560 | 0.9492 | 0.9514 |
| Recall | 0.9487 | 0.9376 | 0.9423 |
| F1-Score | 0.9523 | 0.9433 | 0.9468 |

![Test Results](https://github.com/user-attachments/assets/f523f22a-d63b-4172-a24a-74dbf9639d36)

**Observation:**  
Logistic Regression achieved the best overall performance on the NEP 2020 test set.  
Naive Bayes and SVM also maintained high accuracy (~94%), showing consistent sentiment classification ability on unseen data.  

---

## 📈 Results Summary  

The comparative analysis shows that **Logistic Regression** performed slightly better than SVM and Naive Bayes.  

- Logistic Regression: 87% accuracy  
- SVM: 83% accuracy  
- Naive Bayes: 79% accuracy  

These results confirm the reliability of NLP-based sentiment classification for educational policy analysis.  

---

## 🌐 Streamlit Web Application  

An interactive **Streamlit** web app was developed to make the **NEP 2020 Tweet Sentiment Analyzer** accessible to users in real time.  
It uses the trained **TF-IDF vectorizer** and **Logistic Regression model** for classification.  

🔗 **Live App:** [https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)  

---

## 🧭 How the App Works  

### 📝 Input  
Users can enter one or more tweets related to NEP 2020 — separated by new lines.  

**Example Input:**  
```
NEP 2020 will revolutionize the education system in India.  
The implementation process needs more clarity and transparency.  
I’m still unsure about how NEP 2020 affects higher education.  
```

---

### ⚙️ Processing Pipeline  
- Tweets are cleaned (removal of URLs, emojis, and stopwords).  
- Text is vectorized using the saved TF-IDF model.  
- Each tweet is passed through the trained model for prediction.  
- Streamlit visualizes results in a color-coded table and bar chart.  

---

### 📋 Example Output  

| Tweet | Predicted Sentiment |
| :----- | :----------------- |
| NEP 2020 will revolutionize the education system in India. | Positive |
| The implementation process needs more clarity and transparency. | Negative |
| I’m still unsure about how NEP 2020 affects higher education. | Neutral |

---

## 🖥️ Run the Streamlit App Locally  

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
**Live App:** [Streamlit Link](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)  
**Repository:** Education-Policy-Sentiment  

---

## 📜 License  

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for educational or research purposes, provided proper credit is given.  

---
```

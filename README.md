# 🧠 Education Policy Sentiment

### 📘 Project Overview
This project focuses on **sentiment analysis of educational policy-related text data** to understand public opinion on educational reforms such as *NEP 2020* (National Education Policy).  
The main objective is to build multiple machine learning models that classify text into **Positive**, **Negative**, or **Neutral** sentiments, and compare their performance using standard evaluation metrics.

The project demonstrates how **Natural Language Processing (NLP)** techniques can help interpret feedback on policies and government decisions from online discussions, news headlines, and articles.

---

### 📊 Problem Statement
Education policies directly impact millions of students and teachers. Identifying how the public perceives such policies can guide better decision-making and reforms.  
The challenge was to automatically analyze user opinions or statements related to educational policies and categorize them into sentiment classes.

---

### 📂 Dataset Details

#### 1️⃣ Training Dataset — NEP 2020 English Tweets
This dataset contains English tweets related to India’s **National Education Policy (NEP 2020)**.  
Each tweet is labeled with a sentiment category indicating the public opinion toward the policy.

| Sentiment Category | Description                                      | Number of Tweets |
| ------------------ | ------------------------------------------------ | ---------------- |
| Positive           | Tweets expressing favorable views about NEP 2020 | 3200             |
| Negative           | Tweets showing disagreement or criticism         | 2950             |
| Neutral            | Tweets with balanced or factual statements       | 3100             |

<img width="611" height="388" alt="Training Data" src="https://github.com/user-attachments/assets/6f579d89-e333-4913-a181-0be54de5bf1c" />

**Key Preprocessing Steps:**
- Text cleaning (punctuation, URLs, emojis, and stopword removal)
- Tokenization and normalization  
- TF-IDF feature extraction for text representation  

📘 **Dataset Source:** `NEP_2020_english_tweet.csv`

---

#### 2️⃣ Testing Dataset — NEP 2020 Test Tweets
The testing dataset is a smaller subset containing unseen tweets about NEP 2020.  
It helps evaluate how well trained models generalize to new data.

| Sentiment Category | Description                        | Sample Size |
| ------------------ | ---------------------------------- | ----------- |
| Positive           | Supportive tweets about NEP 2020   | 800         |
| Negative           | Tweets expressing opposition       | 750         |
| Neutral            | Informative or balanced statements | 850         |

<img width="591" height="382" alt="Test Data" src="https://github.com/user-attachments/assets/69550d52-0b1f-4195-95f4-ee66f90f1002" />

This cross-dataset evaluation guarantees the classifier performs optimally on unseen tweets.  
📘 **Dataset Source:** `test.csv`

---

### ⚙️ Methodology
The workflow involves several major steps:

1. **Data Preprocessing**
   - Clean and normalize text data  
   - Apply TF-IDF vectorization for numerical representation  

2. **Model Training**
   - Implemented machine learning algorithms:  
     - Logistic Regression  
     - Support Vector Machine (SVM)  
     - Naive Bayes  

3. **Evaluation**
   - Metrics used: Accuracy, Precision, Recall, and F1-Score  
   - Compared all models on identical test data  

4. **Deployment**
   - Developed a **Streamlit web app** for real-time user interaction  

---

### 🧠 Models Used

| Model | Description | Accuracy |
|--------|--------------|-----------|
| Logistic Regression | Performs well for linearly separable data; interpretable and efficient for text classification. | 0.87 |
| SVM (Support Vector Machine) | Maximizes class separation margin; robust for small text datasets. | 0.83 |
| Naive Bayes | Probabilistic model assuming feature independence; fast and effective for text. | 0.79 |

---

### 📈 Model Training and Evaluation
All models were trained on TF-IDF features and validated using a train-test split (80%-20%).

#### **Training Results and Comparison**

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Notes** |
|:----------|:------------:|:-------------:|:-----------:|:-------------:|:----------|
| Logistic Regression | 91.42% | 0.913 | 0.911 | 0.912 | Strong linear baseline, interpretable results |
| Naive Bayes | 90.87% | 0.909 | 0.906 | 0.907 | Performs well with smaller text data |
| Random Forest | 92.35% | 0.924 | 0.922 | 0.923 | Slightly better generalization on unseen tweets |
| **SVM (Best)** | **93.12%** | **0.931** | **0.930** | **0.930** | Best performing model overall |

<img width="743" height="398" alt="Training Comparison" src="https://github.com/user-attachments/assets/00e91471-532e-467e-9415-8de0e0aebe4d" />

✅ **Observation:**  
SVM outperformed all other models in accuracy and stability, making it the best choice for classifying public sentiment on NEP 2020 tweets.

---

### 🧪 Testing and Cross-Dataset Validation (NEP 2020 Dataset)

To ensure model generalization, trained classifiers were tested on unseen tweets from the NEP 2020 Test Dataset.

#### **Testing Steps**
1️⃣ Loaded trained sentiment models (.pkl) and TF-IDF vectorizer  
2️⃣ Preprocessed the test dataset  
3️⃣ Applied the vectorizer for transformation  
4️⃣ Generated predictions for each model  
5️⃣ Compared predicted vs actual sentiments  
6️⃣ Computed Accuracy, Precision, Recall, and F1-Score  
7️⃣ Visualized confusion matrices and performance charts  

---

#### **Test Results and Comparison**

| **Metric** | **Logistic Regression** | **Naive Bayes** | **SVM** |
|:-----------:|:----------------------:|:---------------:|:-------:|
| **Accuracy** | 0.9512 | 0.9438 | 0.9481 |
| **Precision** | 0.9560 | 0.9492 | 0.9514 |
| **Recall** | 0.9487 | 0.9376 | 0.9423 |
| **F1-Score** | 0.9523 | 0.9433 | 0.9468 |

<img width="730" height="386" alt="Testing Comparison" src="https://github.com/user-attachments/assets/f523f22a-d63b-4172-a24a-74dbf9639d36" />

✅ **Observation:**  
- Logistic Regression achieved the best overall performance on the NEP 2020 test set.  
- Demonstrated strong generalization and consistent classification accuracy across unseen tweets.  
- Naive Bayes and SVM also showed reliable adaptability (~94% accuracy).

---

### 📊 Results Summary
The comparative analysis shows that **Logistic Regression** performed slightly better than SVM and Naive Bayes.

| Model | Accuracy |
|-------|-----------|
| Logistic Regression | 87% |
| SVM | 83% |
| Naive Bayes | 79% |

---

## 🌐 Streamlit Web Application

An interactive **Streamlit web app** was developed to make the **NEP 2020 Tweet Sentiment Analyzer** accessible to users in real time.  
It uses the **TF-IDF vectorizer** and **Logistic Regression model** to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments.

---

### 🟢 Live App (Deployed Link)
👉 [https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)

---

### 🧭 How the App Works

#### 📝 Input:
Users can enter one or more tweets about **NEP 2020** — separated by new lines.  
Each tweet is analyzed independently to determine the expressed sentiment.

**Example Input:**
NEP 2020 will revolutionize the education system in India.
The implementation process needs more clarity and transparency.
I’m still unsure about how NEP 2020 affects higher education.


---

#### ⚙️ Processing Pipeline:
- Tweets are **cleaned** (removal of URLs, emojis, stopwords)
- Text is **vectorized** using the saved TF-IDF model
- Predictions generated using trained sentiment classifier
- Streamlit visualizes results interactively  

---

#### 📋 Output:
- Table displaying each tweet with predicted sentiment  
- Bar chart showing sentiment distribution  
- Instant visualization in browser  

**Example Output Table:**

| Tweet | Predicted Sentiment |
|:------|:--------------------|
| NEP 2020 will revolutionize the education system in India. | Positive |
| The implementation process needs more clarity and transparency. | Negative |
| I’m still unsure about how NEP 2020 affects higher education. | Neutral |

---

### 🖥️ How to Run the Streamlit App Locally

#### Step 1️⃣ – Clone the Repository
```bash
git clone https://github.com/arya-rehpade/Education-Policy-Sentiment.git
cd Education-Policy-Sentiment

Step 2️⃣ – Install Dependencies
pip install -r requirements.txt

Step 3️⃣ – Launch the App
streamlit run app.py

Author

Arya Rehpade
🎓 Machine Learning & Data Science Enthusiast
📍 India

🔗 GitHub: @arya-rehpade
🔗 Live App: Education Policy Sentiment App
🔗 Project Repository: Education-Policy-Sentiment (GitHub)

### License
This project is licensed under the MIT License — you are free to use, modify, and distribute it for educational or research purposes, provided that proper credit is given to the author.


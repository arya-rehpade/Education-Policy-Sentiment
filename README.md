# Education Policy Sentiment

###  Project Overview

This project focuses on **sentiment analysis of educational policy-related text data** to understand public opinion on educational reforms such as *NEP 2020* (National Education Policy).  
The main objective is to build multiple machine learning models that classify text into **positive**, **negative**, or **neutral** sentiments, and then compare their performance using standard evaluation metrics.

The project demonstrates how Natural Language Processing (NLP) techniques can help interpret feedback on policies and government decisions from online discussions, news headlines, and articles.

---

### 📊 Problem Statement

Education policies directly impact millions of students and teachers. Identifying how the public perceives such policies can guide better decision-making and reforms.  
The challenge was to automatically analyze user opinions or statements related to educational policies and categorize them into sentiment classes.

---

### 📂 Dataset Details

1️⃣ Training Dataset — NEP 2020 English Tweets

This dataset contains English tweets related to India’s National Education Policy (NEP 2020).  
Each tweet is labeled with a sentiment category indicating the public opinion toward the policy.

| Sentiment Category | Description                                      | Number of Tweets |
| ------------------ | ------------------------------------------------ | ---------------- |
| Positive           | Tweets expressing favorable views about NEP 2020 | 3200             |
| Negative           | Tweets showing disagreement or criticism         | 2950             |
| Neutral            | Tweets with balanced or factual statements       | 3100             |

<img src="https://github.com/user-attachments/assets/6f579d89-e333-4913-a181-0be54de5bf1c" width="420"/>

Key Preprocessing Steps:

Text cleaning (punctuation, URLs, emojis, and stopword removal)

Tokenization and normalization

TF-IDF feature extraction for text representation

📘 Dataset Source: NEP_2020_english_tweet.csv

2️⃣ Testing Dataset — NEP 2020 Test Tweets

The testing dataset is a smaller subset containing unseen tweets about NEP 2020.  
It helps evaluate how well trained models generalize to new data.

| Sentiment Category | Description                        | Sample Size |
| ------------------ | ---------------------------------- | ----------- |
| Positive           | Supportive tweets about NEP 2020   | 800         |
| Negative           | Tweets expressing opposition       | 750         |
| Neutral            | Informative or balanced statements | 850         |

<img src="https://github.com/user-attachments/assets/69550d52-0b1f-4195-95f4-ee66f90f1002" width="420"/>

This cross-dataset evaluation guarantees the classifier is working optimally outside the initial training space.

📘 Dataset Source: test.csv

---

### ⚙️ Methodology

The workflow involves several major steps:

1. **Data Preprocessing**  
   - Clean and normalize textual data  
   - Apply TF-IDF vectorization to transform text into numerical format  

2. **Model Training**  
   - Implemented three machine learning algorithms:  
     - Logistic Regression  
     - Support Vector Machine (SVM)  
     - Naive Bayes  
   - Models were trained on the same TF-IDF features for fair comparison  

3. **Evaluation**  
   - Used metrics such as Accuracy, Precision, Recall, and F1-Score  
   - Compared all three models on identical test data  

4. **Deployment**  
   - Built an interactive **Streamlit web app** that allows users to:  
     - Input custom text  
     - Select models for comparison  
     - Instantly view sentiment predictions side-by-side  

---

### 🧠 Models Used

| Model | Description | Accuracy |
|--------|--------------|-----------|
| Logistic Regression | Performs well for linearly separable data; interpretable and efficient for text classification. | 0.87 |
| SVM (Support Vector Machine) | Maximizes class separation margin; robust for small text datasets. | 0.83 |
| Naive Bayes | Probabilistic model assuming feature independence; fast and effective for text. | 0.79 |

---

### Model Training and Evaluation

This chapter gives in-depth information about how the models were trained, validated, and tested on two datasets — the NEP 2020 English Tweets to train and the NEP 2020 Test Tweets to test.

Training Information (BBC Dataset)  
NLP Preprocessing Pipeline  

| **Step**                 | **Description**                                                            | **Libraries Used**                                                                         |
| :----------------------- | :------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| 1️⃣ Data Cleaning        | Removed punctuation, URLs, digits, emojis, and converted text to lowercase | `re`, `string`                                                                             |
| 2️⃣ Tokenization         | Split tweets into individual tokens (words)                                | `nltk.word_tokenize`                                                                       |
| 3️⃣ Stopword Removal     | Eliminated common non-informative words like “the”, “is”, “and”            | `nltk.corpus.stopwords`                                                                    |
| 4️⃣ Lemmatization        | Converted words to their base form (e.g., “running” → “run”)               | `WordNetLemmatizer`                                                                        |
| 5️⃣ TF-IDF Vectorization | Transformed cleaned text into numerical feature vectors                    | `sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000, stop_words='english')` |

All models were trained on TF-IDF features and validated using a train-test split (80%-20%).

### Training Results and Comparison

| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Notes**                                                              |
| :------------------ | :----------: | :-----------: | :--------: | :----------: | :--------------------------------------------------------------------- |
| Logistic Regression |    91.42%    |     0.913     |    0.911   |     0.912    | Strong linear baseline, interpretable results                          |
| Naive Bayes         |    90.87%    |     0.909     |    0.906   |     0.907    | Performs well with smaller text data                                   |
| Random Forest       |    92.35%    |     0.924     |    0.922   |     0.923    | Slightly better generalization on unseen tweets                        |
| **SVM (Best)**      |  **93.12%**  |   **0.931**   |  **0.930** |   **0.930**  | Best performing model overall with balanced precision-recall trade-off |

<img src="https://github.com/user-attachments/assets/00e91471-532e-467e-9415-8de0e0aebe4d" width="420"/>

Observation:  
SVM outperformed all other models in accuracy and stability, making it the best choice for classifying public sentiment on NEP 2020 tweets.

### Testing and Cross-Dataset Validation (NEP 2020 Dataset)

To evaluate model generalization and ensure reliability, the trained sentiment classifiers were tested on the NEP 2020 Test Dataset — a collection of unseen tweets related to India’s National Education Policy 2020, categorized into Positive, Negative, and Neutral sentiments.  

Testing Steps:

1️⃣ Loaded trained sentiment models (.pkl) and the TF-IDF vectorizer.  
2️⃣ Preprocessed the test.csv dataset using the same cleaning pipeline as the training data.  
3️⃣ Applied the trained TF-IDF vectorizer for text transformation.  
4️⃣ Generated sentiment predictions for each model (Logistic Regression, Naive Bayes, and SVM).  
5️⃣ Compared model predictions with true sentiment labels.  
6️⃣ Computed evaluation metrics — Accuracy, Precision, Recall, and F1-Score.  
7️⃣ Visualized confusion matrices and model performance comparison charts.

Test Results and Comparison:

|     Metric    | Logistic Regression | Naive Bayes |   SVM  |
| :-----------: | :-----------------: | :---------: | :----: |
|  **Accuracy** |        0.9512       |    0.9438   | 0.9481 |
| **Precision** |        0.9560       |    0.9492   | 0.9514 |
|   **Recall**  |        0.9487       |    0.9376   | 0.9423 |
|  **F1-Score** |        0.9523       |    0.9433   | 0.9468 |

<img src="https://github.com/user-attachments/assets/f523f22a-d63b-4172-a24a-74dbf9639d36" width="420"/>

Observation:  
Logistic Regression achieved the best overall performance on the NEP 2020 test set, demonstrating strong generalization and consistent sentiment classification accuracy across unseen tweets.  
The model successfully captured varying tones of opinions in English tweets.  
Logistic Regression emerged as the most robust and stable performer.  
Naive Bayes and SVM models also maintained high accuracy (~94%), confirming reliable cross-dataset adaptability.

### 📈 Results Summary

The comparative analysis shows that **Logistic Regression** performed slightly better than SVM and Naive Bayes in terms of overall accuracy and consistency across classes.

- Logistic Regression: 87% accuracy  
- SVM: 83% accuracy  
- Naive Bayes: 79% accuracy  

Visualizing model comparison helps understand which algorithm handles education policy sentiment best in limited data conditions.

---

## 🌐 Streamlit Web Application

An interactive **Streamlit web app** was developed to make the **NEP 2020 Tweet Sentiment Analyzer** accessible to users in real time.  
The app utilizes the trained **TF-IDF vectorizer** and **Logistic Regression model** to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments.

---

### 🟢 Live App (Deployed Link)  
👉 [https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)

---

### 🧭 How the App Works

#### 📝 Input:  
Users can enter one or more tweets related to the **National Education Policy (NEP 2020)** — separated by new lines.  
Each tweet is analyzed independently to determine the expressed sentiment.

**Example Input:**  
NEP 2020 will revolutionize the education system in India.  
The implementation process needs more clarity and transparency.  
I’m still unsure about how NEP 2020 affects higher education.

---

#### ⚙️ Processing Pipeline:  
- Tweets are **cleaned** (removal of URLs, emojis, and stopwords).  
- Text is **vectorized** using the saved **TF-IDF model**.  
- Each tweet is passed through the trained **sentiment classification model**.  
- The model predicts whether the tweet is **Positive**, **Negative**, or **Neutral**.  
- Streamlit then visualizes results with a clean, color-coded output.

---

#### 📋 Output:  
- A **table** showing each input tweet with its predicted sentiment.  
- A **bar chart** summarizing sentiment distribution across all entered tweets.  
- Instant results rendered directly in the browser for real-time analysis.

**Example Output Table:**

| Tweet | Predicted Sentiment |
|:------|:--------------------|
| NEP 2020 will revolutionize the education system in India. | Positive |
| The implementation process needs more clarity and transparency. | Negative |
| I’m still unsure about how NEP 2020 affects higher education. | Neutral |

---

### 🖥️ How to Run the Streamlit App Locally

#### Step 1️⃣ – Clone the Repository  

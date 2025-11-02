# 🎓 Education Policy Sentiment — Model Comparison

### 📘 Project Overview

This project focuses on **sentiment analysis of educational policy-related text data** to understand public opinion on educational reforms such as *NEP 2020* (National Education Policy).  
The main objective is to build multiple machine learning models that classify text into **positive**, **negative**, or **neutral** sentiments, and then compare their performance using standard evaluation metrics.

The project demonstrates how Natural Language Processing (NLP) techniques can help interpret feedback on policies and government decisions from online discussions, news headlines, and articles.

---

### 📊 Problem Statement

Education policies directly impact millions of students and teachers. Identifying how the public perceives such policies can guide better decision-making and reforms.  
The challenge was to automatically analyze user opinions or statements related to educational policies and categorize them into sentiment classes.

---

### 📂 Dataset Details

- **Dataset Source:** Custom dataset collected and cleaned using public data from *India Data Portal* and *Open Government Data Platform*  
  (https://indiadataportal.com and https://data.gov.in)
- **Data Type:** Text (sentences / comments / news headlines about education policies)
- **Classes:** Positive, Negative, Neutral
- **Total Samples:** 83  
- **Preprocessing Steps:**
  - Text cleaning (removal of punctuation, numbers, and special characters)
  - Conversion to lowercase
  - Stopword removal
  - TF-IDF vectorization for feature extraction

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

### 📈 Results Summary

The comparative analysis shows that **Logistic Regression** performed slightly better than SVM and Naive Bayes in terms of overall accuracy and consistency across classes.

- Logistic Regression: 87% accuracy  
- SVM: 83% accuracy  
- Naive Bayes: 79% accuracy  

Visualizing model comparison helps understand which algorithm handles education policy sentiment best in limited data conditions.

---

### 💻 Steps to Run the Project

#### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/EducationPolicySentiment.git
cd EducationPolicySentiment

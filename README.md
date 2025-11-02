# Education Policy Sentiment

###  Project Overview

The current project is dedicated to the sentiment analysis of the text data related to the educational policy to get to know what people think about the educational reforms like NEP 2020 (National Education Policy).  
The primary goal is to construct several machine learning models which categorize text and either be positive or negative or neutral, and subsequently evaluate the performance of such models in accordance with the standard evaluation measures.

The project illustrates that Natural Language Processing (NLP) methods may be used to process online discussion feedback concerning policies and government actions on the basis of news headlines and news articles.

---

###  Problem Statement

The policy on education has direct effects on millions of learners and educators. It is possible to identify the way in which the masses view such policies to make better decisions and adjustments.  
The problem was to automatically examine the opinions or statements of the users in the context of educational policies and to classify them according to the sentiment categories.

---

###  Dataset Details

1) Training Data - NEP 2020 English Tweets.

This data consists of tweets in English concerning the National Education Policy (NEP 2020) of India.  
Each of the tweets has a sentiment category that defines how the population views the policy.

| Sentiment Category | Description                                      | Number of Tweets |
| ------------------ | ------------------------------------------------ | ---------------- |
| Positive           | Tweets about NEP 2020 with positive sentiments   | 3200             |
| Negative           | Tweets expressing protest or criticism           | 2950             |
| Neutral            | Tweets that have neutral or factual sentences    | 3100             |

<img src="https://github.com/user-attachments/assets/6f579d89-e333-4913-a181-0be54de5bf1c" width="420"/>

Key Preprocessing Steps:

Text cleaning (punctuations, URLs, emojis and removal of stop words)

Normalization and tokenization.

TF-IDF representational feature extraction of text.

Dataset Source NEP_2020_english_tweet.csv.

2) Testing Dataset — NEP 2020 Test Tweets

The testing dataset is a smaller sample that includes unseen tweets on NEP 2020.  

| Sentiment Category | Description                        | Sample Size |
| ------------------ | ---------------------------------- | ----------- |
| Positive           | Supportive tweets about NEP 2020   | 800         |
| Negative           | Tweets expressing opposition       | 750         |
| Neutral            | Informative or balanced statements | 850         |

<img src="https://github.com/user-attachments/assets/69550d52-0b1f-4195-95f4-ee66f90f1002" width="420"/>

This cross dataset test will ensure that the classifier is operating best beyond the original training space.

 Dataset Source: test.csv

---

###  Methodology

The steps of the workflow include the following significant steps:

1. **Data Preprocessing**  
   - Clean and purify text data.
   - TF-IDF vectorization of text to numerical form.  

2. **Model Training**  
   - Three machine learning algorithms were used:  
     - Logistic Regression  
     - Support Vector Machine (SVM)  
     - Naive Bayes  
   -Fair comparison of models was done using the same TF-IDF features.    

3. **Evaluation**  
   - Accuracy, Precision, Recall and F1-Score were used.    
   - performed a comparison of all three models on the same test data.   

4. **Deployment**  
   - Developed an interactive web value-added application of Streamlit that enables users to:  
     - Input custom text   
     - Choose models to compare them. 
     - See sentiment prediction simultaneously. 

---

###  Models Used

| Model | Description | Accuracy |
|--------|--------------|-----------|
| Logistic Regression | Performs well for linearly separable data; interpretable and efficient for text classification. | 0.87 |
| SVM (Support Vector Machine) | Maximizes class separation margin; robust for small text datasets. | 0.83 |
| Naive Bayes | Probabilistic model assuming feature independence; fast and effective for text. | 0.79 |

---

### Model Training and Evaluation

The chapter provides detailed details on the training, validation and testing of the models on two datasets, namely; the NEP 2020 English Tweets to train the models and the NEP 2020 Test Tweets to test the models.

Training Information (BBC Dataset)  
NLP Preprocessing Pipeline  

| **Step**                 | **Description**                                                            | **Libraries Used**                                                                         |
| :----------------------- | :------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| 1) Data Cleaning        | Removed punctuation, URLs, digits, emojis, and converted text to lowercase | `re`, `string`                                                                             |
| 2) Tokenization         | Split tweets into individual tokens (words)                                | `nltk.word_tokenize`                                                                       |
| 3) Stopword Removal     | Eliminated common non-informative words like “the”, “is”, “and”            | `nltk.corpus.stopwords`                                                                    |
| 4) Lemmatization        | Converted words to their base form (e.g., “running” → “run”)               | `WordNetLemmatizer`                                                                        |
| 5) TF-IDF Vectorization | Transformed cleaned text into numerical feature vectors                    | `sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000, stop_words='english')` |

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
The SVM was more accurate and stable than all other models, so it is the most appropriate to use in classifying the sentiment of the people on NEP 2020 tweets.

### Testing and Cross-Dataset Validation (NEP 2020 Dataset)

In order to gauge model generalization and to guarantee reliability, the trained sentiment classifiers were tested on the NEP 2020 Test Dataset - a set of unseen tweets about the National Education Policy 2020 in India which had Positive, Negative and Neutral sentiments.  

Testing Steps:

  1) Trained sentiment model (.pkl) and the TF-IDF vectorizer.  
  2) Processed the test.csv data with the same cleaning pipeline as with the training data.  
  3) Text transformation with the trained TF-IDF vectorizer.
  4) Predictions of sentiment generated on each of the models (Logistic Regression, naive Bayes and SVM).
  5) Comparison of model predictions and true sentiment labels.    
  6) Calculated metrics of evaluation — Accuracy, Precision, Recall, and F1-Score.  
  7) Visualised confusion matrices and model performance comparison charts.

Test Results and Comparison:

|     Metric    | Logistic Regression | Naive Bayes |   SVM  |
| :-----------: | :-----------------: | :---------: | :----: |
|  **Accuracy** |        0.9512       |    0.9438   | 0.9481 |
| **Precision** |        0.9560       |    0.9492   | 0.9514 |
|   **Recall**  |        0.9487       |    0.9376   | 0.9423 |
|  **F1-Score** |        0.9523       |    0.9433   | 0.9468 |

<img src="https://github.com/user-attachments/assets/f523f22a-d63b-4172-a24a-74dbf9639d36" width="420"/>

Observation:  
The NEP TEAM NEP 2020 test set showed the best overall results on the Logistic Regression, which indicates a good overall generalization and the accuracy of sentiment classification on non-seen tweets.  
The model was able to model the different color shades of opinions in English tweets.  
The strongest and the most stable performer was Logistic Regression.  
High accuracy (94 to 94) was also preserved in Naive Bayes and SVM models proving the credibility of cross-dataset adaptability.

###  Results Summary

The comparative analysis demonstrates that, the overall accuracy and consistency over classes, of the **Logistic Regression worked a little better than SVM and Naive Bayes.

- Logistic Regression: 87% accuracy  
- SVM: 83% accuracy  
- Naive Bayes: 79% accuracy  

The comparison of models is best explored by visualizing so as to understand which algorithm can manage the education policy sentiment under limited data conditions.

---

##  Streamlit Web Application

To provide the NEP 2020 Tweet Sentiment Analyzer to the audience in real time, an interactive web application created with Streamlit was built.  
To identify the sentiment of the tweets, the app makes use of the trained **TF-IDF vectorizer and the Logistic Regression model to classify into either Positive, Negative, or Neutral sentiments.

---

###  Live App (Deployed Link)  
[https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)

---

###  How the App Works

####  Input:  
Users would be allowed to post one or several tweets on the topic of the National Education Policy (NEP 2020) - with new lines between them.  
The analysis of the tweets is done separately in order to identify the expressed sentiment.

** Example Input:**  
The education system in India will be transformed through NEP 2020.  
The process of implementation requires a better understanding and openness.  
I am yet to understand the impact of NEP 2020 on higher education.
---

####  Processing Pipeline:  
- Tweets are cleaned (Url, emojis, and stopwords are removed).    
-The saved TF-IDF model is used to text-vectorize.    
- The trained model of sentiment classification is run on each tweet.   
- The model will give an indication on whether the tweet will be Positive, Negative or Neutral.  
- Streamlit then plots the results in a clean color-coded manner.

---

####  Output:  
- A table containing all the input tweets and the sentiment they are predicted to have.   
- A bar chart that would indicate the distribution of sentiment among all the tweets that have been entered.  
-  Real-time analysis in the form of instant results.

**Example Output Table:**

| Tweet | Predicted Sentiment |
|:------|:--------------------|
| NEP 2020 will revolutionize the education system in India. | Positive |
| The implementation process needs more clarity and transparency. | Negative |
| I’m still unsure about how NEP 2020 affects higher education. | Neutral |

---

###  How to Run the Streamlit App Locally

#### Step 1) – Clone the Repository  
git clone https://github.com/arya-rehpade/Education-Policy-Sentiment.git
cd Education-Policy-Sentiment

#### Step 2) – Install Dependencies
pip install -r requirements.txt

#### Step 3) – Launch the App
streamlit run app.py


---

 Author  
 Arya Rehpade  
 Machine Learning & Data Science Enthusiast  
 India

🔗 GitHub: arya-rehpade  
🔗 Live App: [https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/](https://education-policy-sentiment-bbrz667ksw9zh3xzsfde8p.streamlit.app/)  
🔗 Project Repository: Education-Policy-Sentiment (GitHub)

---

License

This project is licensed under the MIT License — you are free to use, modify, and distribute it for educational or research purposes, provided that proper credit is given to the author.





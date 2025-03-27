# Movie-Review-Analysis
This project classifies movie reviews as Positive or Negative using Natural Language Processing (NLP) techniques and Machine Learning models. The system is built using Logistic Regression with TF-IDF vectorization, leveraging scikit-learn for efficient text classification.
# Getting Started
Follow these steps to set up and run the project on your local machine for development and testing.
## Required Libraries
+ fastapi
+ pickle
+ uvicorn
+ string
+ scikit-learn
+ re
+ numpy
+ BaseModel
+ pandas
# Dataset Used
The dataset consists of thousands of labeled movie reviews, sourced from IMDB. The dataset has two columns:

+ Text: The actual movie review

+ Label: Sentiment classification (Positive or Negative)

We perform data preprocessing, including:
✅ Removing stopwords
✅Tokenization
✅ Lowercasing
# Model Training and Challenges faced
## Model Used
+ Logistic Regression
+ Random Forest
+ SVM (Support Vector Machine)
## Training Process
1 Data Preprocessing:

Tokenization, stopword removal, lowercasing, were applied.

2 Model Training:

The dataset was split (80% train, 20% test).

The model was trained using processed text data.

3️ Evaluation & Optimization:

Hyperparameter tuning was applied for better performance.
## Challenges Faced


🔹 Feature Engineering:

Selecting the right text-processing technique improved accuracy.

🔹 Overfitting Issues:

Early models overfitted due to small datasets.

Regularization & hyperparameter tuning were used to mitigate overfitting.

# Model Performance & Visualizations
To evaluate performance, we use multiple visualizations:
+ Confusion Matrix – Measures accuracy and classification errors
+ Feature Importance Graph – Identifies key words influencing sentiment
+ Word Cloud – Highlights frequently occurring words in Positive/Negative reviews
+ Learning Curves – Helps detect overfitting

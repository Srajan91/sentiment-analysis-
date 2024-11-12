#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle as pk
import os

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r'G:\DHARSHNI_WORKS\Movie_Sentiment_Analysis\IMDB Dataset.csv')

# Drop any null values
df.dropna(inplace=True)

# Clean review function
def clean_review(review):
    return ' '.join(word for word in review.split() if word.lower() not in stopwords.words('english'))

# Apply the clean_review function to the 'review' column
df['review'] = df['review'].apply(clean_review)

# Generate Word Cloud for Negative Reviews
negative_reviews = ' '.join(word for word in df['review'][df['sentiment'] == 'negative'].astype(str))
wordcloud = WordCloud(height=600, width=1000, max_font_size=100).generate(negative_reviews)
plt.figure(figsize=(15, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Negative Reviews")
plt.show()

# Generate Word Cloud for Positive Reviews
positive_reviews = ' '.join(word for word in df['review'][df['sentiment'] == 'positive'].astype(str))
wordcloud = WordCloud(height=600, width=1000, max_font_size=100).generate(positive_reviews)
plt.figure(figsize=(15, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Positive Reviews")
plt.show()

# Vectorize reviews
cv = TfidfVectorizer(max_features=2500)
reviews = cv.fit_transform(df['review']).toarray()

# Encode sentiment labels
df['sentiment'] = df['sentiment'].replace(['positive', 'negative'], [1, 0])

# Split the data into training and testing sets
reviews_train, reviews_test, sent_train, sent_test = train_test_split(reviews, df['sentiment'], test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(reviews_train, sent_train)

# Predict on the test set
predict = model.predict(reviews_test)

# Display the confusion matrix
cm = confusion_matrix(sent_test, predict, labels=model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
display.plot()
plt.show()

# Save the model and vectorizer to files
with open('model.pkl', 'wb') as file:
    pk.dump(model, file)
    pk.dump(cv, open('vectorizer.pkl', 'wb'))

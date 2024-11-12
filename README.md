# Sentiment_Analysis_on_Movie_Review

This project performs sentiment analysis on movie reviews, predicting whether a given review is positive or negative. It uses a Logistic Regression model trained on the IMDB Dataset and a TF-IDF Vectorizer to preprocess and transform the text data. The prediction is displayed using Streamlit with a user-friendly web interface.

**Features:**
Input a movie review and predict whether it is positive or negative.
Display sentiment prediction along with a related image (positive or negative sentiment).
Visualize word clouds for positive and negative reviews from the dataset.

**Technologies Used:**
Python 3.12
Streamlit for the web interface
Scikit-learn for machine learning
Pandas for data manipulation
NLTK for text processing
WordCloud for visualizing word frequency
Matplotlib for plotting
Pickle for saving and loading the trained model

**Project Setup**
**1. Prerequisite**s
Ensure you have Python installed. This project requires the following libraries:

pandas
nltk
scikit-learn
streamlit
matplotlib
wordcloud
pickle
You can install the required libraries using pip:
bash
pip install pandas nltk scikit-learn streamlit matplotlib wordcloud

**2. Model Training**
To use the trained model and vectorizer, follow these steps:

Download the IMDB Dataset (or use your own dataset) and save it as IMDB Dataset.csv.
Preprocess the data:
Clean the reviews by removing stopwords.
Create word clouds for positive and negative reviews.
Train the Logistic Regression model:
Use TfidfVectorizer to convert text into numerical features.
Train a logistic regression classifier to predict sentiments (positive/negative).
Save the trained model and vectorizer:
python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
**3. Running the Web App**
To run the app locally, follow these steps:

Ensure you have the model (model.pkl) and vectorizer (vectorizer.pkl) files saved in the project folder.
Save your images for positive and negative sentiments in the images/ directory (or update the path accordingly).

**Run the Streamlit app using:**
bash
streamlit run app.py
This will open the app in your default web browser.
**4. How to Use**
Once the Streamlit app is running, you'll see a text box to input your movie review.
Type or paste a movie review into the text box.
Press the Predict button to get the sentiment prediction (positive or negative).
The prediction result will be displayed, and an appropriate image representing the sentiment will appear.
**5. Example:**
If the review is "This movie is fantastic, I loved it!", the output would be:
Positive Review
with an image indicating positive sentiment.

If the review is "This movie is terrible, I hated it!", the output would be:
Negative Review
with an image indicating negative sentiment.

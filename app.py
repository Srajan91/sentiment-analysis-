import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model=pk.load(open('model.pkl','rb'))
vectorizer =pk.load(open('vectorizer.pkl','rb'))
review = st.text_input("Enter movie review ")

if st.button('Predict'):
  
    review_scale=vectorizer.transform([review]).toarray()
    result=model.predict(review_scale)

    if result[0]==0:
        st.write('Negative Review')
        st.image( 'G:\\DHARSHNI_WORKS\\Movie_Sentiment_Analysis\\neg.jpg', caption='Negative Sentiment', use_column_width=True)
    else:
         st.write('Postive Review')
         st.image( 'G:\\DHARSHNI_WORKS\\Movie_Sentiment_Analysis\\posi.jpg', caption='Positive Sentiment', use_column_width=True)


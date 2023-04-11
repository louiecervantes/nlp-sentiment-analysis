#Input the relevant libraries
import streamlit as st
import altair as alt
import nltk
import numpy as np
import pandas as pd
import spacy
import random
from textblob import TextBlob

from nltk.tokenize.toktok import ToktokTokenizer
import re
from nltk.classify import accuracy as nltk_accuracy

tokenizer = ToktokTokenizer()



# Define the Streamlit app
def app():
    nlp = []
    if 'en_core_web_sm' in spacy.util.get_installed_models():
        #disable named entity recognizer to reduce memory usage
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    else:
        from spacy.cli import download
        download("en_core_web_sm")

    st.title("TextBlob Sentiment Analysis")      
    st.subheader("(c) 2023 Louie F. Cervantes, M.Eng.")
    
    st.subheader('The TextBlob Package')
    st.write('TextBlob is a Python package for natural language processing (NLP) tasks such as part-of-speech tagging, sentiment analysis, and text classification. It is built on top of the popular Natural Language Toolkit (NLTK) and provides a simple and intuitive API for performing various NLP tasks.')
    
    st.subheader('Simple sentiment analysis task')
    st.write("Sentiment analysis is the process of determining the emotional tone of a piece of text. TextBlob provides two properties for sentiment analysis: polarity and subjectivity. /n/nPolarity refers to the degree to which the text expresses a positive or negative sentiment. Polarity is represented as a float value between -1.0 and 1.0, where -1.0 represents a completely negative sentiment, 0.0 represents a neutral sentiment, and 1.0 represents a completely positive sentiment./n/nSubjectivity, on the other hand, refers to the degree to which the text expresses a subjective or objective viewpoint. Subjectivity is also represented as a float value between 0.0 and 1.0, where 0.0 represents a completely objective viewpoint and 1.0 represents a completely subjective viewpoint.")
    
    st.write('Let us try the following statements.  Copy-paste the statement into the textbox and click the button to get the sentiment.')
    st.write('He is a very good boy.  \
              \nHe is not a good boy. \
              \nEverybody says this man is poor.')
    
    user_input = st.text_input("Input the statement here:")
    
    if st.button('Submit'):  
        result = TextBlob(user_input).sentiment
        st.text(result)
        
    st.subheader('Movie Review Dataset')
    st.write('We load a movie review dataset containing 2 columns: text - contains the text of the review, and label - contains the 0 for negative and 1 for positive reviews. The dataset contains 40,000 rows pf data.')
    
    if st.button('Load Dataset'):  
        df = pd.read_csv('TextBlobTrain.csv')
        st.write(df)
    
# run the app
if __name__ == "__main__":
    app()

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
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    #no and not are excluded from stopwords
    stopwords_list.remove('no')
    stopwords_list.remove('not')
    
    def custom_remove_stopwords(text, is_lower_case=False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
    
    def remove_special_characters(text):
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        return text
    
    def remove_html(text):
        import re
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r' ', text)
    
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r' ', text)
    
    def remove_numbers(text):
        text =''.join([i for i in text if not i.isdigit()])
        return text

    nlp = []
    if 'en_core_web_sm' in spacy.util.get_installed_models():
        #disable named entity recognizer to reduce memory usage
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    else:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm', disable=['ner'])

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
    st.write('We load a movie review dataset containing 2 columns: text - contains the text of the review, and label - contains the 0 for negative and 1 for positive reviews. The dataset contains 40,000 rows pf data. We load the first 20 rows for veiwing.')
    
    if st.button('Load Dataset'):  
        df = pd.read_csv('TextBlobTrain.csv')
        st.write(df.head(20))
        st.write('Dataset shape: ')
        st.text(df.shape)
        
        #Randomly select samples
        label_0=df[df['label']==0].sample(n=5000)
        label_1=df[df['label']==1].sample(n=5000)
        train=pd.concat([label_1, label_0])
        
        from sklearn.utils import shuffle
        train=shuffle(train)
        st.write('We then randomly select 5000 samples of positive reviews and 5000 samples of negative reviews')
        st.write('We display the first 50 rows of the training dataset')
        st.write(train.head(50))
        st.text('Checking for null values')
        st.text(train.isnull().sum())
        
        st.text('Doing pre-processing tasks...')
        st.text('Removing symbols...')
        train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        train.dropna(axis=0, how='any', inplace=True)
        st.text('Removing escape sequences...')
        train.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
        st.text('Removing non ascii data...')
        train['text']=train['text'].str.encode('ascii', 'ignore').str.decode('ascii')
        
        def remove_punctuations(text):
            import string
            for punctuation in string.punctuation:
                text = text.replace(punctuation, '')
            return text
        st.write('Removing punctuations...')
        train['text']=train['text'].apply(remove_punctuations)
        st.write('In Natural Language Processing (NLP), stopwords refer to commonly occurring words in a language that are often filtered out from the text before processing. These words typically do not contribute much to the meaning of a sentence and are used primarily to connect other words together. \nExamples of stopwords in the English language include "the," "a," "an," "and," "in," "on," "at," "for," "to," "of," "with," and so on.')
        st.write('Removing stop words...')
        train['text']=train['text'].apply(custom_remove_stopwords)
        st.write('Removing special characters...')
        train['text']=train['text'].apply(remove_special_characters)
        st.write('Removing HTML...')
        train['text']=train['text'].apply(remove_html)
        st.write('Removing URL...')
        train['text']=train['text'].apply(remove_URL)        
        st.write('Removing numbers...')
        train['text']=train['text'].apply(remove_numbers) 
        st.text('We look at our dataset after the pre-processing steps')
        st.write(train.head(50))
        def cleanse(word):
            rx = re.compile(r'\D*\d')
            if rx.match(word):
                return ''      
            return word
        
        def remove_alphanumeric(strings):
            nstrings= [" ".join(filter(None, (cleanse(word) for word in string.split()))) for string in strings.split()]
            str1=' '.join(nstrings)
            return str1
        st.write('Removing alpha numeric data...')
        train['text']=train['text'].apply(remove_alphanumeric)
        st.text('We look at our dataset after the pre-processing steps')
        st.write(train.head(50))
        
        def lemmatize_text(text):
            text = nlp(text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        
        st.write('We lemmatize the words...')
        train['text']=train['text'].apply(lemmatize_text)
        
        train['sentiment']=train['text'].apply(lambda tweet: TextBlob(tweet).sentiment)
        st.text('We look at our dataset after more pre-processing steps')
        st.write(train.head(50))
        
        sentiment_series=train['sentiment'].tolist()
        columns = ['polarity', 'subjectivity']
        df1 = pd.DataFrame(sentiment_series, columns=columns, index=train.index)
        result = pd.concat([train, df1], axis=1)
        result.drop(['sentiment'],axis=1, inplace=True)
        
        result.loc[result['polarity']>=0.3, 'Sentiment'] = "Positive"
        result.loc[result['polarity']<0.3, 'Sentiment'] = "Negative"
        
        result.loc[result['label']==1, 'Sentiment_label'] = 1
        result.loc[result['label']==0, 'Sentiment_label'] = 0
        
        st.write(result)
        
        counts = result['Sentiment'].value_counts()
        st.write(counts)


# run the app
if __name__ == "__main__":
    app()

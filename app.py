import streamlit as st
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Function to generate word cloud
def generate_wordcloud(text):
    tokens = preprocess_text(text)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
    return wordcloud

# Function for Named Entity Recognition
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI
st.title("📝 NLP Dashboard")
st.subheader("Analyze text for word frequency, sentiment, and named entities.")

# User text input
user_text = st.text_area("Enter text:", "Type here...")

if st.button("Analyze"):
    if user_text:
        # Word Cloud
        st.subheader("Word Frequency & Word Cloud")
        wordcloud = generate_wordcloud(user_text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Named Entity Recognition
        st.subheader("Named Entity Recognition (NER)")
        entities = extract_named_entities(user_text)
        if entities:
            df_entities = pd.DataFrame(entities, columns=["Entity", "Category"])
            st.dataframe(df_entities)
        else:
            st.write("No named entities found.")

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment_result = sentiment_model(user_text)
        st.write(f"**Sentiment:** {sentiment_result[0]['label']} (Score: {sentiment_result[0]['score']:.2f})")

    else:
        st.warning("Please enter some text to analyze.")


import streamlit as st
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Sentiment Analysis Model
sentiment_model = pipeline("sentiment-analysis")

# Function for Text Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    word_freq = Counter(tokens)  # Count word frequency
    return tokens, word_freq

# Function for Generating Word Cloud
def generate_wordcloud(text):
    tokens, _ = preprocess_text(text)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
    return wordcloud

# Function for Named Entity Recognition
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function for POS Tagging
def pos_tagging(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

from gensim import corpora
from gensim.models import LdaModel

def topic_modeling(text, num_topics=3):
    # Tokenize and remove punctuation
    words = [[word for word in word_tokenize(text.lower()) if word.isalnum()]]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [[word for word in doc if word not in stop_words] for doc in words]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Extract topics (ensure only words, not probabilities)
    topics = lda_model.show_topics(num_words=6, formatted=False)  # Get top 6 words per topic
    cleaned_topics = []

    for topic_num, word_list in topics:
        topic_words = [word[0] for word in word_list]  # Extract only words, ignoring probabilities
        cleaned_topics.append((f"Topic {topic_num + 1}", ", ".join(map(str, topic_words))))  # Convert to strings

    return cleaned_topics






# Streamlit UI
st.title("📝 NLP Dashboard")
st.subheader("Analyze text for word frequency, sentiment, named entities, part-of-speech tagging, and topic modeling.")

# User text input
user_text = st.text_area("Enter text:", "Type here...")

if st.button("Analyze"):
    if user_text.strip():  # Ensure text is not empty
        # Word Cloud
        st.subheader("Word Frequency & Word Cloud")
        wordcloud = generate_wordcloud(user_text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Word Frequency Table
        tokens, word_freq = preprocess_text(user_text)
        st.subheader("Top 10 Most Frequent Words")
        freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
        freq_df = freq_df.sort_values(by="Frequency", ascending=False).head(10)
        st.dataframe(freq_df)

        # Named Entity Recognition (NER)
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

        # POS Tagging
        st.subheader("Part-of-Speech (POS) Tagging")
        pos_tags = pos_tagging(user_text)
        df_pos = pd.DataFrame(pos_tags, columns=["Word", "POS"])
        st.dataframe(df_pos)

        # Topic Modeling
        st.subheader("Topic Modeling")
        topics = topic_modeling(user_text)

        if topics:
            for topic in topics:
                st.write(f"**Topic {topic[0]+1}:** {topic[1]}")
        else:
            st.write("No meaningful topics found. Try a longer text.")

    else:
        st.warning("Please enter text before analyzing.")




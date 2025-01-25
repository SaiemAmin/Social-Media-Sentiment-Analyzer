#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import emoji #used to convert emojis into text 
import matplotlib.pyplot as plt
import emoji
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator



# # Pre-Processing the dataset

# In[27]:


df = pd.read_csv("instagram-datasets.csv")
df.shape


# In[28]:


df.head()


# In[29]:


#Converting Emojis to text
df["comment"] = df["comment"].apply(emoji.demojize)

#Dropping redundant data 
df.drop(columns="hashtag_comment", inplace = True)
df.drop(columns = "tagged_users_in_comment", inplace = True)
df.drop(columns = "post_id", inplace = True)


# In[30]:


#Translating text to English using Google Translator API
def translate_text_with_emojis(text, target_lang="en"):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

df["comment"] = df["comment"].apply(translate_text_with_emojis)



# In[31]:


# REMOVING STOPWORDS FROM THE COMMENT COLUMN

nltk.download("stopwords")

# Getting  the list of English stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()  # Split text into words
    # if the word not in stopwords then adding it to the column else not
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Example usage on a DataFrame column
df["comment"] = df["comment"].apply(remove_stopwords)

# Display cleaned comments
df.head()


# In[33]:


#PreProcessing text using Regex

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if len(word) > 2])
    return text

df['comment'] = df['comment'].apply(preprocess_text)


# In[34]:


df.head()


# In[35]:


#Checking NA values for each column
df.isna().sum()


# In[36]:


df.duplicated().sum()


# In[38]:


#Emojizing the comments again

df["comment"] = df["comment"].apply(emoji.emojize)
df.head(5)


# # Applying Hugging Face Pre-trained Model to analyze comment sentiments

# In[39]:


from transformers import pipeline

# Load the sentiment analysis pipeline (using a pre-trained model)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Apply sentiment analysis to Instagram comments
df["sentiment_results"] = df["comment"].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])

# Map labels to more interpretable sentiments
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}
df["sentiment_category"] = df["sentiment_results"].map(label_mapping)

# Display the results
print(df[["comment", "sentiment_category"]])


# # Latent Dirichlet Allocation (LDA)

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import streamlit as st
import pandas as pd

def lda_topic_modeling(df, num_topics=3, max_features=1000):
    """
    Perform LDA topic modeling and coherence scoring on the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'comment' column.
        num_topics (int): Number of topics to extract.
        max_features (int): Maximum number of words for vectorization.

    Returns:
        dict: A dictionary containing topics, coherence score, and feature names.
    """
    
    if "comment" not in df.columns:
        st.error("The dataframe must contain a 'comment' column.")
        return None

    # Vectorize the text data
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(df["comment"])

    # Train LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Extract topics from the LDA model (top 10 words per topic)
    top_words_per_topic = []
    for topic in lda_model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        top_words_per_topic.append(top_words)

    # Convert dataframe column to tokenized list (Gensim input format)
    text_data = [comment.lower().split() for comment in df["comment"]]

    # Create Gensim dictionary and corpus
    dictionary = Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # Convert top words to IDs for Gensim coherence model
    top_words_ids = [[dictionary.token2id[word] for word in topic if word in dictionary.token2id] for topic in top_words_per_topic]

    # Calculate coherence score
    coherence_model = CoherenceModel(topics=top_words_ids, texts=text_data, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    # Return the results
    return {
        "coherence_score": coherence_score,
        "topics": top_words_per_topic,
    
    }

# Step 3: Run the function
results = lda_topic_modeling(df, num_topics=3, max_features=1000)

# Step 4: Print the results
print(f"Coherence Score: {results['coherence_score']:.4f}")
for idx, topic in enumerate(results["topics"]):
    print(f"Topic {idx + 1}: {', '.join(topic)}")



# # Visualization

# In[66]:


def plot_sentiment_distribution(df):
    sentiment_counts = df["sentiment_category"].value_counts()

    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar", color=["blue", "green", "red"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()
plot_sentiment_distribution(df)


# In[ ]:


def sentiment_trend(df):
    
    df['comment_date'] = pd.to_datetime(df['comment_date'])  # Convert to datetime
    df.set_index('comment_date', inplace=True)

    #Resampling the data by day
    sentiment_trend = df.resample("D")["sentiment_category"].value_counts().unstack().fillna(0)

    # Plot the trend
    sentiment_trend.plot(figsize=(10, 6), marker='o')
    plt.title("Sentiment Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.legend(title="Sentiment")
    plt.grid()
    plt.show()

sentiment_trend(df)


# In[43]:


from collections import Counter
from wordcloud import WordCloud

def generate_wordcloud(category):
    words = ' '.join(df[df['sentiment_category'] == category]['comment'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Most Common Words in {category} Comments")
    plt.show()

# Generate word clouds for different sentiment categories
for sentiment in df['sentiment_category'].unique():
    generate_wordcloud(sentiment)


# In[49]:


import pyLDAvis
import pyLDAvis.lda_model as sklearn_vis  # Correct import for sklearn LDA

# Prepare visualization
lda_vis = sklearn_vis.prepare(lda_model, X, vectorizer, mds='tsne')

# Display the visualization in the notebook
pyLDAvis.display(lda_vis)






# In[50]:


# Assuming 'df' is your processed DataFrame
df.to_csv('processed_comments.csv', index=False)

print("Processed data saved successfully as 'processed_comments.csv'")


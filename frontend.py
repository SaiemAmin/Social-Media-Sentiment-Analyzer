import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# Load the processed dataset
@st.cache_data
def load_data():
    df = pd.read_csv("processed_comments.csv")
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    return df

df = load_data()

# Sidebar options for interactive filtering
st.sidebar.title("Filters")

# Date range selection
start_date = st.sidebar.date_input("Start Date", df['comment_date'].min().date())
end_date = st.sidebar.date_input("End Date", df['comment_date'].max().date())

# Sentiment selection
sentiment_options = df["sentiment_category"].unique()
selected_sentiments = st.sidebar.multiselect("Select Sentiments", sentiment_options, default=sentiment_options)

# Keyword search input
search_keyword = st.sidebar.text_input("Search by Keyword", "")

# Apply filters
filtered_df = df[
    (df["comment_date"].dt.date >= start_date) & 
    (df["comment_date"].dt.date <= end_date) & 
    (df["sentiment_category"].isin(selected_sentiments)) &
    (df["comment"].str.contains(search_keyword, case=False, na=False))
]
#Counting the number of sentiments
sentiment_counts = filtered_df["sentiment_category"].value_counts()

# Display filtered results
st.write(f"Showing {len(filtered_df)} comments matching your filters")

# Dashboard Sections
menu = st.sidebar.radio("Select Analysis", ["Overview", "Sentiment Analysis", "Word Clouds", "Comment Search", "LDA visualization"])

# Overview Section
if menu == "Overview":
    st.title("📊 Data Overview")
    st.write(filtered_df.head())
    st.write(f"Total Comments: {len(filtered_df)}")
    
    # Show missing and duplicate values
    st.write("Missing Values:", filtered_df.isna().sum().sum())
    st.write("Duplicate Rows:", filtered_df.duplicated().sum())

    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
    st.plotly_chart(fig)

# Sentiment Analysis Section
elif menu == "Sentiment Analysis":
    st.title("📈 Sentiment Analysis")

    # Sentiment trend over time
    sentiment_trend = filtered_df.resample("D", on="comment_date")["sentiment_category"].value_counts().unstack().fillna(0)
    fig = px.line(sentiment_trend, x=sentiment_trend.index, y=sentiment_trend.columns, title="Sentiment Trend Over Time")
    st.plotly_chart(fig)

    # Sentiment distribution bar chart
    st.subheader("Sentiment Distribution")
    fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, color=sentiment_counts.index, 
                 title="Sentiment Distribution")
    st.plotly_chart(fig)

# Word Cloud Section
elif menu == "Word Clouds":
    st.title("☁️ Word Cloud Visualization")

    # Generate word cloud based on sentiment
    sentiment_option = st.selectbox("Select Sentiment", sentiment_options)
    words = ' '.join(filtered_df[filtered_df['sentiment_category'] == sentiment_option]['comment'])

    #
    if words.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Most Common Words in {sentiment_option} Comments")
        st.pyplot(plt)
    else:
         st.warning(f"No comments available for sentiment: {sentiment_option}. Please try a different sentiment.")

#Latent dirichlet allocation
elif menu == "LDA visualization":
    st.title("Latent Dirchlet  Visualization")

  
    with open("lda_visualization.html", "r", encoding="utf-8") as f:
        lda_html = f.read()
        
    st.success("Showing corpus of topics")
        
         # Render the HTML visualization
    st.components.v1.html(lda_html, width = 1200, height = 500)


# Comment Search Section
elif menu == "Comment Search":
    st.title("🔍 Search Comments")

    # Search and display comments
    st.write("Filtered Comments based on search:")
    st.write(filtered_df[["comment_date", "comment", "sentiment_category"]].sort_values("comment_date", ascending=False))

    # Allow downloading filtered data
    st.download_button(label="Download Filtered Data", 
                       data=filtered_df.to_csv(index=False).encode('utf-8'), 
                       file_name="filtered_comments.csv", 
                       mime="text/csv")





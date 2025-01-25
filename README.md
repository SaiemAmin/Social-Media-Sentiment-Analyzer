# 📊 Social Media Sentiment Analyzer

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
**Live Demo:** [Social Media Sentiment Analyzer](https://social-media-sentiment-analyzer-sbipk74erkb7k6pajmht9y.streamlit.app/)

## 📖 Overview

The **Social Media Sentiment Analyzer** is an interactive web-based dashboard that provides sentiment analysis and topic modeling for social media comments. The project leverages **NLP (Natural Language Processing)** and **machine learning** to classify comments, detect trends, and visualize patterns.

---

## ✨ Features

- **📈 Sentiment Analysis:**  
  - Classifies comments into **Positive, Neutral, or Negative** using a pre-trained **Hugging Face Transformer** model.  
  - Interactive charts displaying sentiment distribution and trends.

- **☁️ Word Cloud Visualization:**  
  - Generate word clouds for different sentiment categories to identify frequently used words.

- **📊 LDA Topic Modeling:**  
  - Discover key topics within the comments using **Latent Dirichlet Allocation (LDA)**.  
  - Interactive topic visualization powered by **pyLDAvis**.

- **📅 Sentiment Trend Over Time:**  
  - Analyze sentiment trends by resampling data over daily/weekly intervals.

- **🔍 Interactive Comment Filtering:**  
  - Search and filter comments by sentiment and keyword.  
  - Export filtered results for further analysis.

---

## 📊 Dashboard Screenshots

| Sentiment Distribution | Word Cloud |
|----------------------|------------|
<img width="1280" alt="Screenshot 2025-01-24 215150" src="https://github.com/user-attachments/assets/68cba16a-b437-4c95-bdc6-df0587b332af" />
<img width="1280" alt="Screenshot 2025-01-24 215217" src="https://github.com/user-attachments/assets/2198eb6c-710a-4301-8ad3-e3fe24a2fc1b" />
<img width="1277" alt="Screenshot 2025-01-24 215246" src="https://github.com/user-attachments/assets/f9bf71e7-93fe-4501-820a-0e392c166db2" />



## 🖥️ Tech Stack

**Frontend:**  
- Streamlit – Web-based visualization framework  
- Plotly – Interactive visualizations  
- Matplotlib – Data visualization  
- WordCloud – Generate word clouds  

**Backend:**  
- Python – Core programming language  
- Pandas – Data analysis and manipulation  
- Scikit-learn – Machine learning and topic modeling  
- Gensim – Topic modeling and coherence scoring  
- Hugging Face Transformers – NLP models  
- PyLDAvis – Topic modeling visualization  

---

## 🚀 How to Run Locally

### **Prerequisites**
- Python 3.8 or above
- Git installed

### **Installation Steps**

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/social-media-sentiment-analyzer.git
    cd social-media-sentiment-analyzer
    ```

2. Create a virtual environment:

    ```bash
    python -m venv env
    source env/bin/activate   # On macOS/Linux
    env\Scripts\activate      # On Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    streamlit run frontend.py
    ```

5. Open the app in your browser at `http://localhost:8501`

---



# 📊 Voices and Views: Tracking Public Pulse on Indian Government Schemes

A comprehensive Big Data pipeline that extracts, analyzes, and visualizes public sentiment about Indian government schemes using news articles and Reddit comments. Built using Apache Spark, Spark NLP, MongoDB, MapReduce, and Apache Pig, the system transforms unstructured text into policy-relevant insights.

<p align="center">
  <img src="./assets/System_architecture.png" alt="System Architecture" width="600">
</p>

---

## 🚀 Project Objectives

- Ingest news and social media data about government schemes
- Clean and process noisy, multilingual text data
- Perform sentiment analysis using deep learning models
- Store and manage processed data for real-time and batch analytics
- Visualize scheme-wise public opinion using dashboards and word clouds

---

## 🛠️ Tech Stack

| Layer           | Tools / Libraries                            |
|-----------------|----------------------------------------------|
| Language        | Python 3.10, PySpark                         |
| Big Data        | Apache Spark 3.4, Hadoop 3.3, Apache Pig     |
| NLP             | Spark NLP 5.1.4, Universal Sentence Encoder  |
| Storage         | MongoDB Atlas, HDFS                          |
| Visualization   | Matplotlib, Seaborn, WordCloud              |
| Social Data     | NewsAPI, Reddit (AsyncPRAW)                  |

---

## 🧠 NLP and Sentiment Pipeline

1. **Preprocessing**: Tokenization, Normalization, Lemmatization, Stopword Removal  
2. **Embedding**: Universal Sentence Encoder (USE)  
3. **Sentiment Classification**: Pretrained deep learning model `sentimentdl_use_twitter` via Spark NLP  
4. **Storage**: MongoDB (cleaned data), HDFS (for batch processing)  
5. **Visualization**: Word clouds, sentiment bar charts, platform comparisons, and timelines

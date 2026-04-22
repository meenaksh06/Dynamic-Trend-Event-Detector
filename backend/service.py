import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

class AnalyticsService:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.topic_trends = None
        self.lda_model = None
        self.vectorizer = None
        self.forecasts = {}
        
        self.load_data()
        self.extract_topics()
        self.generate_trends()
        
    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year_month'] = self.df['date'].dt.to_period('M').astype(str)
        print(f"Data loaded: {len(self.df)} rows")

    def extract_topics(self, n_topics=10):
        print("Extracting topics...")
        self.vectorizer = CountVectorizer(max_features=2000, stop_words='english')
        X = self.vectorizer.fit_transform(self.df['clean_text'].fillna(""))
        
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_dist = self.lda_model.fit_transform(X)
        self.df['dominant_topic'] = topic_dist.argmax(axis=1)
        
        # Get top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_keywords = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            self.topic_keywords.append(top_words)
        print("Topics extracted.")

    def generate_trends(self):
        print("Generating trends...")
        self.topic_trends = self.df.groupby(['year_month', 'dominant_topic']).size().unstack().fillna(0)
        self.topic_trends = self.topic_trends.sort_index()
        print("Trends generated.")

    def get_stats(self):
        return {
            "total_articles": int(len(self.df)),
            "topics_count": int(self.lda_model.n_components),
            "date_range": [str(self.df['date'].min()), str(self.df['date'].max())],
            "top_categories": self.df['category'].value_counts().head(5).to_dict(),
            "sentiment_avg": float(self.df['sentiment_compound'].mean())
        }

    def get_topics(self):
        topics = []
        counts = self.df['dominant_topic'].value_counts().to_dict()
        for i in range(self.lda_model.n_components):
            topics.append({
                "id": i,
                "keywords": self.topic_keywords[i],
                "count": int(counts.get(i, 0)),
                "name": f"Topic {i}: " + ", ".join(self.topic_keywords[i][:3])
            })
        return topics

    def get_trends(self):
        result = []
        for date, row in self.topic_trends.iterrows():
            entry = {"date": date}
            for i, val in enumerate(row):
                entry[f"topic_{i}"] = int(val)
            result.append(entry)
        return result

    def get_forecast(self, topic_id: int):
        # Check cache/pre-calculated if exists, otherwise generate
        if topic_id in self.forecasts:
            return self.forecasts[topic_id]
            
        print(f"Generating forecast for Topic {topic_id}...")
        try:
            data = self.topic_trends.iloc[:, topic_id].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            window_size = 5
            if len(data) <= window_size:
                return {"error": "Not enough data for forecasting"}

            # Simple forward prediction
            X_input = scaled_data[-window_size:].reshape(1, window_size, 1)
            
            # Create a simple LSTM for on-the-fly forecasting if needed
            # In a real production app, we'd load a pre-trained weights file.
            model = Sequential([
                Input(shape=(window_size, 1)),
                LSTM(32, return_sequences=False),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            # Fast train for evaluation demo (10 epochs)
            model.fit(X_input, scaled_data[-1:], epochs=10, verbose=0)
            
            # Predict 6 months ahead
            current_batch = scaled_data[-window_size:].reshape(1, window_size, 1)
            future_preds = []
            
            for _ in range(6):
                pred = model.predict(current_batch, verbose=0)[0]
                future_preds.append(pred[0])
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                
            future_unscaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            
            forecast_data = []
            last_date = datetime.strptime(self.topic_trends.index[-1], "%Y-%m")
            for i, val in enumerate(future_unscaled):
                new_month = (last_date.month + i) % 12 + 1
                new_year = last_date.year + (last_date.month + i) // 12
                date_str = f"{new_year}-{new_month:02d}"
                forecast_data.append({"date": date_str, "value": float(val)})
                
            self.forecasts[topic_id] = forecast_data
            return forecast_data
        except Exception as e:
            print(f"Forecasting error: {e}")
            return []

    def get_articles(self, topic_id: int, limit: int = 10):
        subset = self.df[self.df['dominant_topic'] == topic_id].sort_values('date', ascending=False).head(limit)
        return subset[['date', 'headline', 'category', 'sentiment_compound', 'link']].to_dict(orient='records')

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timezone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


class AnalyticsService:
    """
    Core analytics engine for the Dynamic Trend & Event Detector.
    Hybrid forecasting: ARIMA (linear) + MLP Neural Network (non-linear residuals).
    """
    WINDOW_SIZE = 5
    N_TOPICS = 10

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.DataFrame()
        self.topic_trends = pd.DataFrame()
        self.lda_model = None
        self.vectorizer = None
        self.forecasts = {}
        self.ablation_cache = {}
        self.topic_keywords = []
        self.live_event_buffer = []
        self.load_data()
        self.extract_topics()
        self.generate_trends()

    def load_data(self):
        print("[1/3] Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year_month'] = self.df['date'].dt.to_period('M').astype(str)
        print(f"      Loaded {len(self.df)} articles")

    def extract_topics(self):
        print("[2/3] Extracting topics via LDA...")
        self.vectorizer = CountVectorizer(max_features=2000, stop_words='english')
        X = self.vectorizer.fit_transform(self.df['clean_text'].fillna(""))
        self.lda_model = LatentDirichletAllocation(n_components=self.N_TOPICS, random_state=42)
        topic_dist = self.lda_model.fit_transform(X)
        self.df['dominant_topic'] = topic_dist.argmax(axis=1)
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_keywords = []
        for topic in self.lda_model.components_:
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            self.topic_keywords.append(top_words)
        print("      Topics extracted.")

    def generate_trends(self):
        print("[3/3] Generating trends...")
        self.topic_trends = (
            self.df.groupby(['year_month', 'dominant_topic']).size().unstack().fillna(0)
        )
        self.topic_trends = self.topic_trends.sort_index()
        print("      Trends ready. Server starting...\n")

    # ── Helpers ──

    @staticmethod
    def _create_sequences(data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i + window])
            y.append(data[i + window])
        return np.array(X), np.array(y)

    @staticmethod
    def _evaluate(actual, predicted):
        return {
            'RMSE': round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            'MAE': round(float(mean_absolute_error(actual, predicted)), 4),
            'R2': round(float(r2_score(actual, predicted)), 4),
            'MAPE': round(float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100), 2),
        }

    def _build_nn(self):
        """Neural network using sklearn MLP — avoids TF mutex issues on macOS."""
        return MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )

    def _nn_forecast(self, train_data, n_forecast, scaler):
        """Train neural net on sequences and forecast n steps ahead."""
        X, y = self._create_sequences(train_data, self.WINDOW_SIZE)
        if len(X) < 3:
            return np.zeros(n_forecast)

        model = self._build_nn()
        model.fit(X, y)

        window = train_data[-self.WINDOW_SIZE:].copy()
        preds = []
        for _ in range(n_forecast):
            p = model.predict(window.reshape(1, -1))[0]
            preds.append(p)
            window = np.append(window[1:], p)

        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # ══════════════════════════════════════════════════════
    # API ENDPOINTS
    # ══════════════════════════════════════════════════════

    def get_stats(self):
        return {
            "total_articles": len(self.df),
            "topics_count": getattr(self.lda_model, 'n_components', self.N_TOPICS),
            "date_range": [str(self.df['date'].min()), str(self.df['date'].max())],
            "top_categories": self.df['category'].value_counts().head(5).to_dict(),
            "sentiment_avg": float(self.df['sentiment_compound'].mean()),
            "hybrid_model": "ARIMA(2,1,2) + NeuralNet(64-32-16)",
        }

    def get_topics(self):
        topics = []
        counts = self.df['dominant_topic'].value_counts().to_dict()
        num_topics = getattr(self.lda_model, 'n_components', self.N_TOPICS)
        for i in range(num_topics):
            topics.append({
                "id": i,
                "keywords": self.topic_keywords[i] if i < len(self.topic_keywords) else [],
                "count": int(counts.get(i, 0)),
                "name": f"Topic {i}: " + ", ".join(self.topic_keywords[i][:3]) if i < len(self.topic_keywords) else f"Topic {i}"
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

    def get_sentiment_timeline(self):
        monthly = (
            self.df.groupby('year_month')['sentiment_compound']
            .agg(['mean', 'std', 'count']).reset_index()
        )
        monthly.columns = ['date', 'avg_sentiment', 'std_sentiment', 'article_count']
        monthly = monthly.sort_values('date')
        return monthly.to_dict(orient='records')

    def get_forecast(self, topic_id: int):
        """Legacy basic forecast endpoint."""
        if topic_id in self.forecasts:
            return self.forecasts[topic_id]

        try:
            data = self.topic_trends.iloc[:, topic_id].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data).flatten()

            if len(data) <= self.WINDOW_SIZE:
                return {"error": "Not enough data"}

            X, y = self._create_sequences(scaled, self.WINDOW_SIZE)
            model = self._build_nn()
            model.fit(X, y)

            window = scaled[-self.WINDOW_SIZE:].copy()
            future_preds = []
            for _ in range(6):
                pred = model.predict(window.reshape(1, -1))[0]
                future_preds.append(pred)
                window = np.append(window[1:], pred)

            future_vals = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            last_date = datetime.strptime(self.topic_trends.index[-1], "%Y-%m")
            forecast_data = []
            for i, val in enumerate(future_vals):
                new_month = (last_date.month + i) % 12 + 1
                new_year = last_date.year + (last_date.month + i) // 12
                forecast_data.append({"date": f"{new_year}-{new_month:02d}", "value": float(val)})

            self.forecasts[topic_id] = forecast_data
            return forecast_data
        except Exception as e:
            print(f"Forecast error: {e}")
            return []

    def get_hybrid_forecast(self, topic_id: int):
        """
        Phase 3 Hybrid: ARIMA + Neural Network residual correction.
        Returns all 3 model predictions for comparison.
        """
        cache_key = f"hybrid_{topic_id}"
        if cache_key in self.forecasts:
            return self.forecasts[cache_key]

        print(f"  → Hybrid forecast: Topic {topic_id}...")
        try:
            series = self.topic_trends.iloc[:, topic_id].values.astype(float)
            split_idx = int(len(series) * 0.8)
            train, test = series[:split_idx], series[split_idx:]

            if len(test) < 2 or len(train) < self.WINDOW_SIZE + 2:
                return {"error": "Not enough data"}

            dates = list(self.topic_trends.index)
            train_dates = dates[:split_idx]
            test_dates = dates[split_idx:]

            # ── A) ARIMA (ML Component) ──
            future_steps = 6
            total_steps = len(test) + future_steps

            arima_model = ARIMA(train, order=(2, 1, 2))
            arima_fit = arima_model.fit()
            arima_pred_all = np.array(arima_fit.forecast(steps=total_steps))
            arima_pred = arima_pred_all[:len(test)]
            future_arima = arima_pred_all[len(test):]
            
            arima_residuals = train[1:] - np.array(arima_fit.fittedvalues)[1:]

            # ── B) Neural Net on raw data (DL-Only) ──
            scaler_raw = MinMaxScaler()
            scaled_train = scaler_raw.fit_transform(train.reshape(-1, 1)).flatten()
            dl_forecast_all = self._nn_forecast(scaled_train, total_steps, scaler_raw)
            dl_forecast = dl_forecast_all[:len(test)]
            future_bilstm = dl_forecast_all[len(test):]

            # ── C) Hybrid: ARIMA + Neural Net on residuals ──
            scaler_res = MinMaxScaler(feature_range=(-1, 1))
            scaled_res = scaler_res.fit_transform(arima_residuals.reshape(-1, 1)).flatten()

            if len(scaled_res) > self.WINDOW_SIZE:
                res_corrections_all = self._nn_forecast(scaled_res, total_steps, scaler_res)
                hybrid_pred_all = arima_pred_all + res_corrections_all
                hybrid_pred = hybrid_pred_all[:len(test)]
                future_hybrid = hybrid_pred_all[len(test):]
            else:
                hybrid_pred = arima_pred.copy()
                future_hybrid = future_arima.copy()

            # ── Metrics ──
            arima_metrics = self._evaluate(test, arima_pred)
            dl_metrics = self._evaluate(test, dl_forecast)
            hybrid_metrics = self._evaluate(test, hybrid_pred)

            # Generating future dates
            last_date = datetime.strptime(self.topic_trends.index[-1], "%Y-%m")
            future_dates = []
            for i in range(1, future_steps + 1):
                new_month = (last_date.month + i - 1) % 12 + 1
                new_year = last_date.year + (last_date.month + i - 1) // 12
                future_dates.append(f"{new_year}-{new_month:02d}")

            result = {
                "topic_id": topic_id,
                "split_index": split_idx,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "future_dates": future_dates,
                "actual": test.tolist(),
                "train_values": train.tolist(),
                "arima": {"predictions": arima_pred.tolist(), "metrics": arima_metrics},
                "bilstm": {"predictions": dl_forecast.tolist(), "metrics": dl_metrics},
                "hybrid": {"predictions": hybrid_pred.tolist(), "metrics": hybrid_metrics},
                "future_arima": future_arima.tolist(),
                "future_bilstm": future_bilstm.tolist(),
                "future_hybrid": future_hybrid.tolist(),
            }
            self.forecasts[cache_key] = result
            print(f"    RMSE → ARIMA:{arima_metrics['RMSE']}, NN:{dl_metrics['RMSE']}, Hybrid:{hybrid_metrics['RMSE']}")
            return result
        except Exception as e:
            print(f"Hybrid error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_ablation(self):
        """Run ablation study across topics 0-4."""
        if self.ablation_cache:
            return self.ablation_cache

        print("  → Running ablation study...")
        topics_to_test = list(range(min(5, self.topic_trends.shape[1])))
        all_results = []

        for tid in topics_to_test:
            result = self.get_hybrid_forecast(tid)
            if not isinstance(result, dict) or "error" in result:
                continue
            for model_name, key in [("ARIMA (ML)", "arima"), ("BiLSTM (DL)", "bilstm"), ("Hybrid", "hybrid")]:
                row = {"topic": tid, "model": model_name}
                if key in result and isinstance(result[key], dict) and "metrics" in result[key]:
                    metrics_data = result[key]["metrics"]
                    if isinstance(metrics_data, dict):
                        row.update(metrics_data) # type: ignore
                all_results.append(row)

        averages, diagnostics = [], []
        if all_results:
            df_res = pd.DataFrame(all_results)
            avg = df_res.groupby('model')[['RMSE', 'MAE', 'R2', 'MAPE']].mean()
            for model_name, row in avg.iterrows():
                averages.append({
                    "model": model_name, "RMSE": round(row['RMSE'], 4),
                    "MAE": round(row['MAE'], 4), "R2": round(row['R2'], 4),
                    "MAPE": round(row['MAPE'], 2),
                })
            hybrid_rmse = avg.loc['Hybrid']['RMSE'] if 'Hybrid' in avg.index else 1
            if 'ARIMA (ML)' in avg.index:
                d = ((avg.loc['ARIMA (ML)']['RMSE'] - hybrid_rmse) / hybrid_rmse) * 100
                diagnostics.append({
                    "component_removed": "Neural Net (DL)",
                    "impact": f"RMSE changes by {d:+.1f}%",
                    "explanation": "ARIMA alone cannot capture non-linear event-driven spikes."
                })
            if 'BiLSTM (DL)' in avg.index:
                d = ((avg.loc['BiLSTM (DL)']['RMSE'] - hybrid_rmse) / hybrid_rmse) * 100
                diagnostics.append({
                    "component_removed": "ARIMA (ML)",
                    "impact": f"RMSE changes by {d:+.1f}%",
                    "explanation": "Neural Net alone overfits on short sequences, missing linear trends."
                })

        self.ablation_cache = {
            "per_topic": all_results, "averages": averages,
            "diagnostics": diagnostics, "topics_tested": len(topics_to_test),
        }
        print("    Ablation complete.")
        return self.ablation_cache

    def get_articles(self, topic_id: int, limit: int = 10):
        subset = (
            self.df[self.df['dominant_topic'] == topic_id]
            .sort_values('date', ascending=False).head(limit)
        )
        return subset[['date', 'headline', 'category', 'sentiment_compound', 'link']].to_dict(orient='records')

    # ══════════════════════════════════════════════════════
    # LIVE DATA (WebSocket)
    # ══════════════════════════════════════════════════════

    _EVENT_TYPES = ["breaking", "trending", "update", "update", "update"]

    def generate_live_event(self):
        """Sample a random article from the dataset to simulate a live incoming event."""
        row = self.df.sample(1).iloc[0]
        topic_id = int(row.get('dominant_topic', 0))
        topic_name = (
            f"Topic {topic_id}: " + ", ".join(self.topic_keywords[topic_id][:3])
            if topic_id < len(self.topic_keywords)
            else f"Topic {topic_id}"
        )
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "headline": str(row.get('headline', '')),
            "category": str(row.get('category', '')),
            "sentiment_compound": float(row.get('sentiment_compound', 0.0)),
            "topic_id": topic_id,
            "topic_name": topic_name,
            "link": str(row.get('link', '')),
            "event_type": random.choice(self._EVENT_TYPES),
            "word_count": int(row.get('word_count', 0)),
        }
        # Buffer last 50 events
        self.live_event_buffer.append(event)
        if len(self.live_event_buffer) > 50:
            self.live_event_buffer = self.live_event_buffer[-50:]
        return event

    def get_live_stats(self):
        """Return buffered events and rolling stats for the live page initial load."""
        events = self.live_event_buffer[-20:]
        topic_activity = {}
        sentiments = []
        for ev in self.live_event_buffer:
            tid = ev.get('topic_id', 0)
            topic_activity[tid] = topic_activity.get(tid, 0) + 1
            sentiments.append(ev.get('sentiment_compound', 0.0))
        avg_sent = float(np.mean(sentiments)) if sentiments else 0.0
        return {
            "events": events,
            "topic_activity": topic_activity,
            "avg_sentiment": round(avg_sent, 4),
            "total_events": len(self.live_event_buffer),
        }

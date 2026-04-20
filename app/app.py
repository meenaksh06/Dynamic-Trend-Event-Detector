import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("📊 Dynamic Topic Trend Forecasting")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/processed_featured_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    return df

df = load_data()

# Create topic trends (simulate if not present)
if 'dominant_topic' not in df.columns:
    st.warning("dominant_topic not found. Using random topics for demo.")
    df['dominant_topic'] = np.random.randint(0, 5, size=len(df))

topic_trends = df.groupby(
    ['year_month', 'dominant_topic']
).size().unstack().fillna(0)

topic_trends = topic_trends.sort_index()

# UI
st.sidebar.header("Controls")

topic_index = st.sidebar.slider(
    "Select Topic",
    0,
    topic_trends.shape[1] - 1,
    0
)

# Plot
st.subheader(f"📈 Topic {topic_index} Trend")

fig, ax = plt.subplots()
ax.plot(topic_trends.iloc[:, topic_index])
ax.set_xlabel("Time")
ax.set_ylabel("Count")
ax.set_title("Topic Trend Over Time")

st.pyplot(fig)

# Info
st.markdown("""
### 🔍 About
- Phase 1: Topic extraction (LDA / BERTopic)
- Phase 2: Forecasting using LSTM
- Regularization: Dropout, EarlyStopping, L2
""")
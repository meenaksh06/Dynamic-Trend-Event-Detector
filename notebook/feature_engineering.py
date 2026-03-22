import pandas as pd
from tqdm import tqdm
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data for VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

tqdm.pandas()

input_file = '/Users/meenakshsinghania04/Desktop/Dynamic-Trend-Event-Detector/data/processed/processed_news.csv'
output_dir = '/Users/meenakshsinghania04/Desktop/Dynamic-Trend-Event-Detector/data/processed'
output_file = os.path.join(output_dir, 'featured_news.csv')

def main():
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['clean_text'] = df['clean_text'].fillna('')
    df['date'] = pd.to_datetime(df['date'])

    print("Extracting temporal features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    print("Extracting text length features...")
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['clean_text'].apply(lambda x: len(str(x)))

    print("Extracting sentiment features using VADER...")
    sia = SentimentIntensityAnalyzer()
    
    def get_vader_scores(text):
        if not text.strip():
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        return sia.polarity_scores(text)
        
    scores = df['clean_text'].progress_apply(get_vader_scores)
    scores_df = pd.DataFrame(scores.tolist())
    
    for col in scores_df.columns:
        df[f'sentiment_{col}'] = scores_df[col]

    print(f"Saving featured data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Feature engineering completed successfully!")
    print("\nSample of engineered features:")
    print(df[['year', 'word_count', 'sentiment_compound']].head())

if __name__ == '__main__':
    main()

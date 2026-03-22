import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if not exists
os.makedirs('project-reports/images', exist_ok=True)

# Try loading the data
try:
    df = pd.read_csv('data/processed/processed_featured_data.csv')
    
    # 1. Category Distribution (Top 15)
    plt.figure(figsize=(10, 6))
    top_categories = df['category'].value_counts().head(15)
    sns.barplot(y=top_categories.index, x=top_categories.values, palette='viridis')
    plt.title('Top 15 News Categories', fontsize=14)
    plt.xlabel('Number of Articles', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('project-reports/images/category_distribution.png', dpi=300)
    plt.close()

    # 2. Articles per Year (Temporal Trend)
    if 'year' in df.columns:
        plt.figure(figsize=(10, 5))
        year_counts = df['year'].value_counts().sort_index()
        sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', linewidth=2)
        plt.title('Article Volume Over Time (Years)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Articles', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Ensure integers on x axis
        plt.xticks(year_counts.index.astype(int))
        plt.tight_layout()
        plt.savefig('project-reports/images/temporal_trend.png', dpi=300)
        plt.close()
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        plt.figure(figsize=(10, 5))
        year_counts = df['year'].value_counts().sort_index()
        sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', linewidth=2)
        plt.title('Article Volume Over Time (Years)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(year_counts.index.astype(int))
        plt.tight_layout()
        plt.savefig('project-reports/images/temporal_trend.png', dpi=300)
        plt.close()

    # 3. Word Count Distribution
    if 'word_count' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['word_count'], bins=50, kde=True, color='purple')
        plt.title('Word Count Distribution of Articles', fontsize=14)
        plt.xlabel('Word Count', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        # Limit x axis to avoid long tail skewing visualization
        plt.xlim(0, df['word_count'].quantile(0.99))
        plt.tight_layout()
        plt.savefig('project-reports/images/word_count_distribution.png', dpi=300)
        plt.close()

    print("Success: Visualizations generated in project-reports/images")
except Exception as e:
    print(f"Error: {e}")

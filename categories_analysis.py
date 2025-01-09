import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Create 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def calculate_gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    # Sort array
    sorted_array = np.sort(x)
    n = len(x)
    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_array)
    # Calculate Gini coefficient
    return ((n + 1) / n) - (2 / (n * np.mean(x) * n)) * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_array)

# Read the CSV file
df = pd.read_csv('data/temporal_patterns.csv')

# Convert month to datetime for better handling
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

# 1. Top Categories Analysis
def plot_top_categories():
    category_totals = df.groupby('category')['duration'].sum().sort_values(ascending=True)
    
    # Calculate Gini coefficient
    gini = calculate_gini(category_totals.values)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(category_totals)), category_totals.values)
    plt.yticks(range(len(category_totals)), category_totals.index)
    
    # Add value labels
    for i, v in enumerate(category_totals.values):
        plt.text(v, i, f' {v:.1f}h', va='center')
    
    plt.title(f'Total Duration by Category\nGini Coefficient: {gini:.3f}')
    plt.xlabel('Total Duration (hours)')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('results/category_totals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_totals, gini

# 2. Generate Summary Statistics
def generate_summary(category_totals, gini):
    summary = {
        'total_duration': df['duration'].sum(),
        'avg_monthly_duration': df.groupby('month')['duration'].sum().mean(),
        'peak_month': df.groupby('month')['duration'].sum().idxmax(),
        'top_category': df.groupby('category')['duration'].sum().idxmax(),
        'unique_categories': df['category'].nunique(),
        'gini_coefficient': gini,
        'top_20_percent_share': (np.sum(np.sort(category_totals.values)[-int(len(category_totals)*0.2):]) / 
                                np.sum(category_totals.values) * 100)
    }
    
    # Save summary to text file
    with open('results/category_analysis_summary.txt', 'w') as f:
        f.write("Category Analysis Summary\n")
        f.write("=======================\n\n")
        f.write(f"Total Duration: {summary['total_duration']:.2f} hours\n")
        f.write(f"Average Monthly Duration: {summary['avg_monthly_duration']:.2f} hours\n")
        f.write(f"Peak Month: {summary['peak_month'].strftime('%Y-%m')}\n")
        f.write(f"Top Category: {summary['top_category']}\n")
        f.write(f"Number of Unique Categories: {summary['unique_categories']}\n")
        f.write(f"Gini Coefficient: {summary['gini_coefficient']:.3f}\n")
        f.write(f"Top 20% Categories Share: {summary['top_20_percent_share']:.1f}%\n")

def main():
    print("Starting category analysis...")
    
    # Generate visualization and get category totals and Gini coefficient
    category_totals, gini = plot_top_categories()
    print(f"Top categories visualization completed (Gini coefficient: {gini:.3f})")
    
    generate_summary(category_totals, gini)
    print("Summary statistics generated")
    
    print("Analysis complete! All results saved in the 'results' folder.")

if __name__ == "__main__":
    main()
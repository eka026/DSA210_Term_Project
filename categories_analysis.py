import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Read the CSV file
df = pd.read_csv('data/temporal_patterns.csv')

# Convert month to datetime for better handling
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

# 1. Top Categories Analysis
def plot_top_categories():
    category_totals = df.groupby('category')['duration'].sum().sort_values(ascending=True)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(category_totals)), category_totals.values)
    plt.yticks(range(len(category_totals)), category_totals.index)
    
    # Add value labels
    for i, v in enumerate(category_totals.values):
        plt.text(v, i, f' {v:.1f}h', va='center')
    
    plt.title('Total Duration by Category')
    plt.xlabel('Total Duration (hours)')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('results/category_totals.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Generate Summary Statistics
def generate_summary():
    summary = {
        'total_duration': df['duration'].sum(),
        'avg_monthly_duration': df.groupby('month')['duration'].sum().mean(),
        'peak_month': df.groupby('month')['duration'].sum().idxmax(),
        'top_category': df.groupby('category')['duration'].sum().idxmax(),
        'unique_categories': df['category'].nunique()
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

def main():
    print("Starting category analysis...")
    
    # Generate visualization
    plot_top_categories()
    print("Top categories visualization completed")
    
    generate_summary()
    print("Summary statistics generated")
    
    print("Analysis complete! All results saved in the 'results' folder.")

if __name__ == "__main__":
    main()
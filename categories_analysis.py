import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# 2. Subcategory Distribution with merged "other" categories
def plot_subcategory_distribution():
    # Calculate total duration
    subcategory_totals = df.groupby('subcategory')['duration'].sum()
    total_duration = subcategory_totals.sum()
    
    # Calculate percentages
    subcategory_percentages = (subcategory_totals / total_duration) * 100
    
    # Define threshold for 'Other' category
    threshold = 1.0
    
    # Create plot data dictionary
    plot_data = {}
    other_total = 0
    
    # Process each subcategory
    for subcat, percentage in subcategory_percentages.items():
        if percentage >= threshold and subcat.lower() != 'other':
            plot_data[subcat] = percentage
        else:
            # Add to other_total for both small percentages and existing 'other' category
            other_total += percentage
    
    # Add the combined 'Other' category
    if other_total > 0:
        plot_data['Other'] = other_total
    
    # Convert to Series and sort
    plot_data = pd.Series(plot_data).sort_values(ascending=False)
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(plot_data)), plot_data.values)
    plt.xticks(range(len(plot_data)), plot_data.index, rotation=45, ha='right')
    
    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Distribution of Viewing Duration by Subcategory')
    plt.xlabel('Subcategory')
    plt.ylabel('Percentage of Total Duration')
    plt.tight_layout()
    plt.savefig('results/subcategory_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Monthly Activity Heatmap
def plot_monthly_heatmap():
    # Create pivot table for heatmap
    monthly_activity = df.pivot_table(
        values='duration',
        index=df['month'].dt.strftime('%Y'),
        columns=df['month'].dt.strftime('%m'),
        aggfunc='sum'
    )
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(monthly_activity, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Monthly Activity Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig('results/monthly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Generate Summary Statistics
def generate_summary():
    summary = {
        'total_duration': df['duration'].sum(),
        'avg_monthly_duration': df.groupby('month')['duration'].sum().mean(),
        'peak_month': df.groupby('month')['duration'].sum().idxmax(),
        'top_category': df.groupby('category')['duration'].sum().idxmax(),
        'unique_categories': df['category'].nunique(),
        'unique_subcategories': df['subcategory'].nunique()
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
        f.write(f"Number of Unique Subcategories: {summary['unique_subcategories']}\n")

def main():
    print("Starting category analysis...")
    
    # Generate visualizations
    plot_top_categories()
    print("Top categories visualization completed")
    
    plot_subcategory_distribution()
    print("Subcategory distribution visualization completed")
    
    plot_monthly_heatmap()
    print("Monthly heatmap completed")
    
    generate_summary()
    print("Summary statistics generated")
    
    print("Analysis complete! All results saved in the 'results' folder.")

if __name__ == "__main__":
    main()
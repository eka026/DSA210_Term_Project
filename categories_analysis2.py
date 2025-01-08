import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def validate_data(enhanced_df, channel_df):
    """Validate and clean the data"""
    print("\nPerforming data validation...")
    
    # Get sets of channels from both dataframes
    enhanced_channels = set(enhanced_df['channel'].unique())
    channel_df_channels = set(channel_df['channel'].unique())
    
    # Find channels that are in enhanced_df but not in channel_df
    suspicious_channels = enhanced_channels - channel_df_channels
    if suspicious_channels:
        print(f"\nFound {len(suspicious_channels)} channels in enhanced_categories.csv that don't appear in channel_categories.csv")
        print("These will be excluded from the analysis.")
    
    # Filter out suspicious channels
    validated_df = enhanced_df[enhanced_df['channel'].isin(channel_df_channels)].copy()
    
    # Remove entries with unusually high durations (e.g., more than 24 hours)
    duration_outliers = validated_df[validated_df['duration'] > 24]
    if len(duration_outliers) > 0:
        print(f"\nFound {len(duration_outliers)} entries with durations > 24 hours")
        print("These will be excluded from the analysis.")
        validated_df = validated_df[validated_df['duration'] <= 24]
    
    # Calculate channel statistics
    channel_stats = validated_df.groupby('channel').agg({
        'duration': ['count', 'sum'],
        'watched_on': 'nunique'
    }).reset_index()
    channel_stats.columns = ['channel', 'video_count', 'total_hours', 'unique_days']
    
    # Filter out channels with very low engagement (e.g., only one video or very short duration)
    min_videos = 2  # Minimum number of videos
    min_duration = 0.5  # Minimum total hours
    valid_channels = channel_stats[
        (channel_stats['video_count'] >= min_videos) & 
        (channel_stats['total_hours'] >= min_duration)
    ]['channel']
    
    validated_df = validated_df[validated_df['channel'].isin(valid_channels)]
    
    print(f"\nAfter validation:")
    print(f"Original entries: {len(enhanced_df)}")
    print(f"Validated entries: {len(validated_df)}")
    print(f"Original unique channels: {len(enhanced_channels)}")
    print(f"Validated unique channels: {len(validated_df['channel'].unique())}")
    
    return validated_df

def analyze_peak_period(peak_data, period_name, results_dir):
    """Analyze a specific peak period"""
    
    # Calculate total hours and view count separately for subcategories
    subcategory_hours = peak_data.groupby(['month', 'category', 'subcategory'])['duration'].sum().reset_index(name='total_hours')
    subcategory_views = peak_data.groupby(['month', 'category', 'subcategory']).size().reset_index(name='view_count')
    
    # Merge the results
    subcategory_analysis = pd.merge(subcategory_hours, subcategory_views, 
                                  on=['month', 'category', 'subcategory'])
    
    # Calculate percentage within each month
    monthly_totals = subcategory_analysis.groupby('month')['total_hours'].sum()
    subcategory_analysis['percentage'] = subcategory_analysis.apply(
        lambda x: (x['total_hours'] / monthly_totals[x['month']]) * 100, axis=1
    )
    
    # Channel analysis with minimum thresholds
    channel_hours = peak_data.groupby(['month', 'channel', 'category'])['duration'].sum().reset_index(name='total_hours')
    channel_views = peak_data.groupby(['month', 'channel', 'category']).size().reset_index(name='view_count')
    channel_analysis = pd.merge(channel_hours, channel_views, on=['month', 'channel', 'category'])
    
    # Filter out channels with very low engagement within the period
    channel_analysis = channel_analysis[
        (channel_analysis['total_hours'] >= 0.5) &  # At least 30 minutes
        (channel_analysis['view_count'] >= 2)       # At least 2 views
    ]
    
    # Sort results
    top_subcategories = subcategory_analysis.sort_values(['month', 'total_hours'], ascending=[True, False])
    top_channels = channel_analysis.sort_values(['month', 'total_hours'], ascending=[True, False])
    
    # Save analysis results
    with open(os.path.join(results_dir, f'{period_name}_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== {period_name} Analysis ===\n\n")
        
        for month in sorted(peak_data['month'].unique()):
            f.write(f"\n{month} Analysis:\n")
            f.write("-" * 50 + "\n")
            
            # Overall statistics
            monthly_data = peak_data[peak_data['month'] == month]
            f.write(f"\nOverall Statistics:\n")
            f.write(f"Total Hours: {monthly_data['duration'].sum():.1f}\n")
            f.write(f"Unique Channels: {len(monthly_data['channel'].unique())}\n")
            f.write(f"Total Views: {len(monthly_data)}\n")
            
            # Top 10 subcategories
            f.write("\nTop 10 Subcategories:\n")
            month_subcats = top_subcategories[top_subcategories['month'] == month].head(10)
            for _, row in month_subcats.iterrows():
                f.write(f"{row['category']} - {row['subcategory']}:\n")
                f.write(f"  Hours: {row['total_hours']:.1f} ({row['percentage']:.1f}%)\n")
                f.write(f"  Views: {row['view_count']}\n")
            
            # Top 10 channels
            f.write("\nTop 10 Channels:\n")
            month_channels = top_channels[top_channels['month'] == month].head(10)
            for _, row in month_channels.iterrows():
                f.write(f"{row['channel']} ({row['category']}):\n")
                f.write(f"  Hours: {row['total_hours']:.1f}\n")
                f.write(f"  Views: {row['view_count']}\n")
            
            # Additional validation info
            f.write("\nValidation Info:\n")
            f.write(f"Channels filtered out: {len(peak_data[peak_data['month'] == month]['channel'].unique()) - len(month_channels['channel'].unique())}\n")
    
    # Create visualizations
    plt.style.use('ggplot')
    
    # 1. Category Distribution
    plt.figure(figsize=(15, 10))
    
    # Calculate category totals
    category_analysis = peak_data.groupby(['month', 'category'])['duration'].sum().reset_index(name='total_hours')
    
    # Calculate percentage for each category
    category_totals = category_analysis.groupby('month')['total_hours'].sum()
    category_analysis['percentage'] = category_analysis.apply(
        lambda x: (x['total_hours'] / category_totals[x['month']]) * 100, axis=1
    )
    
    # Sort categories by total hours
    top_categories = category_analysis.groupby('category')['total_hours'].sum().sort_values(ascending=False)
    category_analysis['category'] = pd.Categorical(
        category_analysis['category'], 
        categories=top_categories.index, 
        ordered=True
    )
    
    # Create the plot
    ax = sns.barplot(data=category_analysis, x='total_hours', y='category', hue='month',
                    palette=sns.color_palette('husl', n_colors=len(category_analysis['month'].unique())))
    plt.title(f'Category Distribution - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Category', size=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{period_name}_categories.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Channel Distribution
    plt.figure(figsize=(15, 10))
    top_10_channels = channel_analysis.groupby('channel')['total_hours'].sum()\
                     .sort_values(ascending=False).head(10)
    
    channel_plot_data = channel_analysis[channel_analysis['channel'].isin(top_10_channels.index)]
    
    ax = sns.barplot(data=channel_plot_data, x='total_hours', y='channel', hue='month',
                    palette=sns.color_palette('husl', n_colors=len(channel_plot_data['month'].unique())))
    plt.title(f'Top 10 Channels - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Channel', size=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{period_name}_top_channels.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_peaks():
    # Create results directory
    results_dir = 'results'
    create_directory(results_dir)
    
    # Read the CSV files
    enhanced_df = pd.read_csv('data/enhanced_categories.csv')
    channel_df = pd.read_csv('data/channel_categories.csv')
    
    # Validate the data
    validated_df = validate_data(enhanced_df, channel_df)
    
    # Convert watched_on to datetime using ISO format
    validated_df['watched_on'] = pd.to_datetime(validated_df['watched_on'], format='ISO8601')
    validated_df['month'] = validated_df['watched_on'].dt.strftime('%Y-%m')
    validated_df['duration'] = pd.to_numeric(validated_df['duration'], errors='coerce')
    
    # Define peak periods
    peak_periods = {
        '2021_May_June': ['2021-05', '2021-06'],
        '2022_May_June': ['2022-05', '2022-06'],
        '2023_September': ['2023-09'],
        '2024_September': ['2024-09']
    }
    
    # Analyze each peak period
    for period_name, months in peak_periods.items():
        print(f"\nAnalyzing {period_name}...")
        peak_data = validated_df[validated_df['month'].isin(months)].copy()
        analyze_peak_period(peak_data, period_name, results_dir)
    
    print("\nAnalysis complete. Results saved in results/")

if __name__ == "__main__":
    analyze_all_peaks()
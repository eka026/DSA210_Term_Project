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

def analyze_peak_period(peak_data, period_name, results_dir, aggregate=False):
    """Analyze a specific peak period"""
    
    # Calculate total hours and view count separately for subcategories
    group_cols = ['category', 'subcategory'] if aggregate else ['month', 'category', 'subcategory']
    
    subcategory_hours = peak_data.groupby(group_cols)['duration'].sum().reset_index(name='total_hours')
    subcategory_views = peak_data.groupby(group_cols).size().reset_index(name='view_count')
    
    # Merge the results
    subcategory_analysis = pd.merge(subcategory_hours, subcategory_views, 
                                  on=group_cols)
    
    # Calculate percentage
    if aggregate:
        total_hours = subcategory_analysis['total_hours'].sum()
        subcategory_analysis['percentage'] = (subcategory_analysis['total_hours'] / total_hours) * 100
    else:
        monthly_totals = subcategory_analysis.groupby('month')['total_hours'].sum().to_dict()
        subcategory_analysis['percentage'] = subcategory_analysis.apply(
            lambda x: (x['total_hours'] / monthly_totals[x['month']]) * 100 if not aggregate else 0, 
            axis=1
        )
    
    # Channel analysis with minimum thresholds
    group_cols = ['channel', 'category'] if aggregate else ['month', 'channel', 'category']
    channel_hours = peak_data.groupby(group_cols)['duration'].sum().reset_index(name='total_hours')
    channel_views = peak_data.groupby(group_cols).size().reset_index(name='view_count')
    channel_analysis = pd.merge(channel_hours, channel_views, on=group_cols)
    
    # Filter out channels with very low engagement within the period
    channel_analysis = channel_analysis[
        (channel_analysis['total_hours'] >= 0.5) &  # At least 30 minutes
        (channel_analysis['view_count'] >= 2)       # At least 2 views
    ]
    
    # Sort results
    if aggregate:
        top_subcategories = subcategory_analysis.sort_values('total_hours', ascending=False)
        top_channels = channel_analysis.sort_values('total_hours', ascending=False)
    else:
        top_subcategories = subcategory_analysis.sort_values(['month', 'total_hours'], ascending=[True, False])
        top_channels = channel_analysis.sort_values(['month', 'total_hours'], ascending=[True, False])
    
    # Save analysis results
    with open(os.path.join(results_dir, f'{period_name}_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== {period_name} Analysis ===\n\n")
        
        if aggregate:
            f.write("\nOverall Period Analysis:\n")
            f.write("-" * 50 + "\n")
            
            # Overall statistics
            f.write(f"\nOverall Statistics:\n")
            f.write(f"Total Hours: {peak_data['duration'].sum():.1f}\n")
            f.write(f"Unique Channels: {len(peak_data['channel'].unique())}\n")
            f.write(f"Total Views: {len(peak_data)}\n")
            f.write(f"Period Coverage: {min(peak_data['month'])} to {max(peak_data['month'])}\n")
            
            # Top 10 subcategories
            f.write("\nTop 10 Subcategories:\n")
            for _, row in top_subcategories.head(10).iterrows():
                f.write(f"{row['category']} - {row['subcategory']}:\n")
                f.write(f"  Hours: {row['total_hours']:.1f} ({row['percentage']:.1f}%)\n")
                f.write(f"  Views: {row['view_count']}\n")
            
            # Top 10 channels
            f.write("\nTop 10 Channels:\n")
            for _, row in top_channels.head(10).iterrows():
                f.write(f"{row['channel']} ({row['category']}):\n")
                f.write(f"  Hours: {row['total_hours']:.1f}\n")
                f.write(f"  Views: {row['view_count']}\n")
        else:
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
    
    # Create visualizations
    plt.style.use('ggplot')
    
    # 1. Category Distribution
    plt.figure(figsize=(15, 10))
    
    # Calculate category totals
    if aggregate:
        category_analysis = peak_data.groupby(['category'])['duration'].sum().reset_index(name='total_hours')
        total_hours = category_analysis['total_hours'].sum()
        category_analysis['percentage'] = (category_analysis['total_hours'] / total_hours) * 100
    else:
        category_analysis = peak_data.groupby(['month', 'category'])['duration'].sum().reset_index(name='total_hours')
        category_totals = category_analysis.groupby('month')['total_hours'].sum()
        category_analysis['percentage'] = category_analysis.apply(
            lambda x: (x['total_hours'] / category_totals[x['month']]) * 100, axis=1
        )
    
    # Filter out categories with very low hours (less than 1 hour total)
    if aggregate:
        category_analysis = category_analysis[category_analysis['total_hours'] >= 1]
    else:
        # Calculate total hours per category across all months
        category_totals = category_analysis.groupby('category')['total_hours'].sum()
        valid_categories = category_totals[category_totals >= 1].index
        category_analysis = category_analysis[category_analysis['category'].isin(valid_categories)]
    
    # Sort categories by total hours
    top_categories = category_analysis.groupby('category')['total_hours'].sum().sort_values(ascending=False)
    category_analysis['category'] = pd.Categorical(
        category_analysis['category'], 
        categories=top_categories.index, 
        ordered=True
    )
    
    # Create the plot with improved styling
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create the horizontal bar plot
    if aggregate:
        bars = sns.barplot(data=category_analysis, 
                          x='total_hours', 
                          y='category',
                          hue='category',
                          palette='husl',
                          legend=False,
                          errorbar=None,
                          ax=ax)
    else:
        # Create color palette for categories
        unique_categories = category_analysis['category'].unique()
        category_colors = dict(zip(unique_categories, sns.husl_palette(n_colors=len(unique_categories))))
        
        # Plot each month's data separately
        positions = range(len(unique_categories))
        width = 0.35  # Width of the bars
        
        months = sorted(category_analysis['month'].unique())
        for i, month in enumerate(months):
            month_data = category_analysis[category_analysis['month'] == month]
            offset = (i - len(months)/2 + 0.5) * width  # Center the bars
            
            # Plot bars without error bars
            bars = ax.barh(y=[positions[list(unique_categories).index(cat)] + offset 
                            for cat in month_data['category']],
                          width=month_data['total_hours'],
                          height=width,
                          label=month,
                          color=[category_colors[cat] for cat in month_data['category']])
        
        # Adjust y-axis ticks
        ax.set_yticks(positions)
        ax.set_yticklabels(unique_categories)
    
    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Customize labels
    plt.title(f'Category Distribution - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Category', size=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{period_name}_categories.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Channel Distribution
    plt.figure(figsize=(15, 10))
    top_10_channels = channel_analysis.groupby('channel')['total_hours'].sum()\
                     .sort_values(ascending=False).head(10)
    
    channel_plot_data = channel_analysis[channel_analysis['channel'].isin(top_10_channels.index)]
    
    # Create channel plot with similar styling
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create the horizontal bar plot
    if aggregate:
        bars = sns.barplot(data=channel_plot_data, 
                          x='total_hours', 
                          y='channel',
                          hue='channel',
                          palette='husl',
                          legend=False,
                          errorbar=None,
                          ax=ax)
    else:
        # Create color palette for channels
        unique_channels = channel_plot_data['channel'].unique()
        channel_colors = dict(zip(unique_channels, sns.husl_palette(n_colors=len(unique_channels))))
        
        # Plot each month's data separately
        positions = range(len(unique_channels))
        width = 0.35  # Width of the bars
        
        months = sorted(channel_plot_data['month'].unique())
        for i, month in enumerate(months):
            month_data = channel_plot_data[channel_plot_data['month'] == month]
            offset = (i - len(months)/2 + 0.5) * width  # Center the bars
            
            # Plot bars without error bars
            bars = ax.barh(y=[positions[list(unique_channels).index(ch)] + offset 
                            for ch in month_data['channel']],
                          width=month_data['total_hours'],
                          height=width,
                          label=month,
                          color=[channel_colors[ch] for ch in month_data['channel']])
        
        # Adjust y-axis ticks
        ax.set_yticks(positions)
        ax.set_yticklabels(unique_channels)
    
    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Customize labels
    plt.title(f'Top 10 Channels - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Channel', size=12)
    
    # Adjust layout and save
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
        'Gaming_Peak_2020_2021': {
            'months': ['2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', 
                      '2020-11', '2020-12', '2021-01', '2021-02'],
            'aggregate': True  # Flag to indicate this period should be aggregated
        },
        '2021_May_June': {
            'months': ['2021-05', '2021-06'],
            'aggregate': True
        },
        '2022_May_June': {
            'months': ['2022-05', '2022-06'],
            'aggregate': True
        },
        '2023_September': {
            'months': ['2023-09'],
            'aggregate': True
        },
        '2024_September': {
            'months': ['2024-09'],
            'aggregate': True
        }
    }
    
    # Analyze each peak period
    for period_name, period_info in peak_periods.items():
        print(f"\nAnalyzing {period_name}...")
        peak_data = validated_df[validated_df['month'].isin(period_info['months'])].copy()
        analyze_peak_period(peak_data, period_name, results_dir, aggregate=period_info['aggregate'])

if __name__ == "__main__":
    analyze_all_peaks()
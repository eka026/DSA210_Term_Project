import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from scipy import stats

###############################################################################
#                         PART 1: CATEGORY ANALYSIS                           #
#            (Analyzes data from 'data/temporal_patterns.csv')               #
###############################################################################

# Create 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def calculate_gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    sorted_array = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_array)
    return ((n + 1) / n) - (2 / (n * np.mean(x) * n)) * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_array)

# Read the CSV file for the category analysis
# Change path as needed.
df = pd.read_csv('data/temporal_patterns.csv')

# Convert month to datetime for better handling
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

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

def generate_summary(category_totals, gini):
    summary = {
        'total_duration': df['duration'].sum(),
        'avg_monthly_duration': df.groupby('month')['duration'].sum().mean(),
        'peak_month': df.groupby('month')['duration'].sum().idxmax(),
        'top_category': df.groupby('category')['duration'].sum().idxmax(),
        'unique_categories': df['category'].nunique(),
        'gini_coefficient': gini,
        'top_20_percent_share': (
            np.sum(np.sort(category_totals.values)[-int(len(category_totals)*0.2):]) / 
            np.sum(category_totals.values) * 100
        )
    }
    
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

def category_analysis_main():
    """ Main function for the first script (Category Analysis). """
    print("Starting category analysis...")
    category_totals, gini = plot_top_categories()
    print(f"Top categories visualization completed (Gini coefficient: {gini:.3f})")
    
    generate_summary(category_totals, gini)
    print("Summary statistics generated")
    
    print("Analysis complete! All results saved in the 'results' folder.")


###############################################################################
#                       PART 2: CHANNEL/PEAK ANALYSIS                         #
#       (Analyzes data from 'data/enhanced_categories.csv' &                  #
#                   'data/channel_categories.csv')                            #
###############################################################################

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def validate_data(enhanced_df, channel_df):
    """Validate and clean the data"""
    print("\nPerforming data validation...")
    
    enhanced_channels = set(enhanced_df['channel'].unique())
    channel_df_channels = set(channel_df['channel'].unique())
    
    # Channels in enhanced_df but not in channel_df
    suspicious_channels = enhanced_channels - channel_df_channels
    if suspicious_channels:
        print(f"\nFound {len(suspicious_channels)} channels in enhanced_categories.csv "
              f"that don't appear in channel_categories.csv")
        print("These will be excluded from the analysis.")
    
    # Filter out suspicious channels
    validated_df = enhanced_df[enhanced_df['channel'].isin(channel_df_channels)].copy()
    
    # Remove entries with unusually high durations (> 24 hours)
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
    
    # Filter out low engagement channels
    min_videos = 2   # Minimum number of videos
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
    """
    Analyze a specific peak period:
      - Summaries by category/subcategory
      - Summaries by channel
      - Plots stored in results_dir
    """
    group_cols = ['category', 'subcategory'] if aggregate else ['month', 'category', 'subcategory']
    
    # Subcategory stats
    subcategory_hours = peak_data.groupby(group_cols)['duration'].sum().reset_index(name='total_hours')
    subcategory_views = peak_data.groupby(group_cols).size().reset_index(name='view_count')
    
    subcategory_analysis = pd.merge(subcategory_hours, subcategory_views, on=group_cols)
    
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
    
    # Channel analysis
    group_cols = ['channel', 'category'] if aggregate else ['month', 'channel', 'category']
    channel_hours = peak_data.groupby(group_cols)['duration'].sum().reset_index(name='total_hours')
    channel_views = peak_data.groupby(group_cols).size().reset_index(name='view_count')
    channel_analysis = pd.merge(channel_hours, channel_views, on=group_cols)
    
    # Filter out channels with very low engagement in the period
    channel_analysis = channel_analysis[
        (channel_analysis['total_hours'] >= 0.5) &
        (channel_analysis['view_count'] >= 2)
    ]
    
    if aggregate:
        top_subcategories = subcategory_analysis.sort_values('total_hours', ascending=False)
        top_channels = channel_analysis.sort_values('total_hours', ascending=False)
    else:
        top_subcategories = subcategory_analysis.sort_values(['month', 'total_hours'],
                                                             ascending=[True, False])
        top_channels = channel_analysis.sort_values(['month', 'total_hours'],
                                                    ascending=[True, False])
    
    # Save analysis text results
    with open(os.path.join(results_dir, f'{period_name}_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== {period_name} Analysis ===\n\n")
        
        if aggregate:
            f.write("\nOverall Period Analysis:\n")
            f.write("-" * 50 + "\n")
            
            # Stats
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
    
    # Filter categories with very low hours
    if aggregate:
        category_analysis = category_analysis[category_analysis['total_hours'] >= 1]
    else:
        cat_totals = category_analysis.groupby('category')['total_hours'].sum()
        valid_categories = cat_totals[cat_totals >= 1].index
        category_analysis = category_analysis[category_analysis['category'].isin(valid_categories)]
    
    # Sort categories by total hours
    top_categories = category_analysis.groupby('category')['total_hours'].sum().sort_values(ascending=False)
    category_analysis['category'] = pd.Categorical(
        category_analysis['category'],
        categories=top_categories.index,
        ordered=True
    )
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    if aggregate:
        sns.barplot(
            data=category_analysis, 
            x='total_hours', 
            y='category',
            hue='category',
            palette='husl',
            legend=False,
            errorbar=None,
            ax=ax
        )
    else:
        # Color palette for categories
        unique_categories = category_analysis['category'].unique()
        category_colors = dict(zip(unique_categories, sns.husl_palette(n_colors=len(unique_categories))))
        
        positions = range(len(unique_categories))
        width = 0.35
        months = sorted(category_analysis['month'].unique())
        
        for i, month in enumerate(months):
            month_data = category_analysis[category_analysis['month'] == month]
            offset = (i - len(months)/2 + 0.5) * width
            ax.barh(
                y=[positions[list(unique_categories).index(cat)] + offset
                   for cat in month_data['category']],
                width=month_data['total_hours'],
                height=width,
                label=month,
                color=[category_colors[cat] for cat in month_data['category']]
            )
        
        ax.set_yticks(positions)
        ax.set_yticklabels(unique_categories)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Category Distribution - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Category', size=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{period_name}_categories.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Channel Distribution
    plt.figure(figsize=(15, 10))
    top_10_channels = channel_analysis.groupby('channel')['total_hours'].sum().sort_values(ascending=False).head(10)
    channel_plot_data = channel_analysis[channel_analysis['channel'].isin(top_10_channels.index)]
    
    fig, ax = plt.subplots(figsize=(15, 10))
    if aggregate:
        sns.barplot(
            data=channel_plot_data, 
            x='total_hours', 
            y='channel',
            hue='channel',
            palette='husl',
            legend=False,
            errorbar=None,
            ax=ax
        )
    else:
        unique_channels = channel_plot_data['channel'].unique()
        channel_colors = dict(zip(unique_channels, sns.husl_palette(n_colors=len(unique_channels))))
        
        positions = range(len(unique_channels))
        width = 0.35
        months = sorted(channel_plot_data['month'].unique())
        
        for i, month in enumerate(months):
            month_data = channel_plot_data[channel_plot_data['month'] == month]
            offset = (i - len(months)/2 + 0.5) * width
            ax.barh(
                y=[positions[list(unique_channels).index(ch)] + offset
                   for ch in month_data['channel']],
                width=month_data['total_hours'],
                height=width,
                label=month,
                color=[channel_colors[ch] for ch in month_data['channel']]
            )
        
        ax.set_yticks(positions)
        ax.set_yticklabels(unique_channels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Top 10 Channels - {period_name}', pad=20, size=14)
    plt.xlabel('Hours Watched', size=12)
    plt.ylabel('Channel', size=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{period_name}_top_channels.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_peaks():
    results_dir = 'results'
    create_directory(results_dir)
    
    # Read the CSV files
    enhanced_df = pd.read_csv('data/enhanced_categories.csv')
    channel_df = pd.read_csv('data/channel_categories.csv')
    
    # Validate data
    validated_df = validate_data(enhanced_df, channel_df)
    
    # Convert watched_on to datetime
    validated_df['watched_on'] = pd.to_datetime(validated_df['watched_on'], format='ISO8601')
    validated_df['month'] = validated_df['watched_on'].dt.strftime('%Y-%m')
    validated_df['duration'] = pd.to_numeric(validated_df['duration'], errors='coerce')
    
    # Define peak periods
    peak_periods = {
        'Gaming_Peak_2020_2021': {
            'months': [
                '2020-05','2020-06','2020-07','2020-08','2020-09','2020-10',
                '2020-11','2020-12','2021-01','2021-02'
            ],
            'aggregate': True
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
    
    # Analyze each peak
    for period_name, period_info in peak_periods.items():
        print(f"\nAnalyzing {period_name}...")
        peak_data = validated_df[validated_df['month'].isin(period_info['months'])].copy()
        analyze_peak_period(peak_data, period_name, results_dir, aggregate=period_info['aggregate'])


###############################################################################
#                      PART 3: HYPOTHESIS TEST FUNCTIONS                      #
#             (Requires 'enhanced_categories.csv' for analysis)              #
###############################################################################

def perform_gaming_hypothesis_test(data, results_dir):
    """
    Perform hypothesis test comparing gaming vs non-gaming watch time
    for the Gaming Peak period (2020-2021)
    """
    gaming_data = data[data['category'] == 'Gaming']['duration']
    nongaming_data = data[data['category'] != 'Gaming']['duration']
    
    gaming_stats = {
        'n': len(gaming_data),
        'mean': gaming_data.mean(),
        'median': gaming_data.median(),
        'std': gaming_data.std(),
    }
    
    nongaming_stats = {
        'n': len(nongaming_data),
        'mean': nongaming_data.mean(),
        'median': nongaming_data.median(),
        'std': nongaming_data.std(),
    }
    
    # One-tailed Mann-Whitney U (gaming > non-gaming)
    statistic, p_value = stats.mannwhitneyu(gaming_data, nongaming_data, alternative='greater')
    
    output_file = os.path.join(results_dir, 'gaming_peak_hypothesis_test.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Gaming Peak Period (2020-2021) Hypothesis Test ===\n")
        f.write("\nPeriod Coverage: ")
        f.write(f"{min(data['month'])} to {max(data['month'])}\n")
        
        f.write("\nHypothesis Test Results:")
        f.write("\n" + "="*50)
        f.write("\nNull Hypothesis (H₀): No difference in watch time between gaming and non-gaming content")
        f.write("\nAlternative Hypothesis (H₁): Gaming content has higher watch time than non-gaming content")
        
        f.write("\n\nDescriptive Statistics:")
        f.write("\n\nGaming Content:")
        f.write(f"\nn = {gaming_stats['n']}")
        f.write(f"\nMean = {gaming_stats['mean']:.3f} hours")
        f.write(f"\nMedian = {gaming_stats['median']:.3f} hours")
        f.write(f"\nStd Dev = {gaming_stats['std']:.3f} hours")
        
        f.write("\n\nNon-Gaming Content:")
        f.write(f"\nn = {nongaming_stats['n']}")
        f.write(f"\nMean = {nongaming_stats['mean']:.3f} hours")
        f.write(f"\nMedian = {nongaming_stats['median']:.3f} hours")
        f.write(f"\nStd Dev = {nongaming_stats['std']:.3f} hours")
        
        f.write("\n\nMann-Whitney U Test Results:")
        f.write(f"\nU-statistic: {statistic}")
        f.write(f"\np-value: {p_value:.6f}")
        
        alpha = 0.05
        f.write(f"\n\nAt α = {alpha}:")
        if p_value < alpha:
            f.write("\nReject H₀: There is strong statistical evidence that gaming content")
            f.write("\nhas higher watch time than non-gaming content (p < 0.05)")
        else:
            f.write("\nFail to reject H₀: There is not enough statistical evidence that gaming")
            f.write("\ncontent has higher watch time than non-gaming content (p ≥ 0.05)")

def perform_education_shift_hypothesis_test(data_2021, data_2022, results_dir):
    """
    Perform z-test for comparing educational content proportions
    between May-June 2021 and May-June 2022
    """
    def get_education_stats(data):
        education_data = data[data['category'] == 'Education']['duration']
        noneducation_data = data[data['category'] != 'Education']['duration']
        
        n_education = len(education_data)
        n_total = len(education_data) + len(noneducation_data)
        proportion = n_education / n_total if n_total > 0 else 0
        
        def format_duration(hours):
            hours_part = int(hours)
            minutes_part = int((hours - hours_part) * 60)
            return f"{hours_part:02d}:{minutes_part:02d}"
        
        def calculate_duration_stats(durations):
            if len(durations) == 0:
                return {"mean": 0, "median": 0, "distribution": {}}
            
            mean_hours = durations.mean()
            median_hours = durations.median()
            
            distribution = {
                "< 15min": len(durations[durations <= 0.25]) / len(durations) * 100,
                "15-30min": len(durations[(durations > 0.25) & (durations <= 0.5)]) * 100 / len(durations),
                "30-60min": len(durations[(durations > 0.5) & (durations <= 1.0)]) * 100 / len(durations),
                "> 1hour": len(durations[durations > 1.0]) * 100 / len(durations)
            }
            
            return {
                "mean": mean_hours,
                "median": median_hours,
                "formatted_mean": format_duration(mean_hours),
                "formatted_median": format_duration(median_hours),
                "distribution": distribution
            }
        
        edu_stats = calculate_duration_stats(education_data)
        
        return {
            'n_education': n_education,
            'n_total': n_total,
            'proportion': proportion,
            'education_hours': education_data.sum(),
            'total_hours': data['duration'].sum(),
            'mean_duration': edu_stats['mean'],
            'median_duration': edu_stats['median'],
            'formatted_mean': edu_stats['formatted_mean'],
            'formatted_median': edu_stats['formatted_median'],
            'duration_distribution': edu_stats['distribution']
        }
    
    stats_2021 = get_education_stats(data_2021)
    stats_2022 = get_education_stats(data_2022)
    
    # Two-proportion z-test
    p1 = stats_2021['proportion']
    p2 = stats_2022['proportion']
    n1 = stats_2021['n_total']
    n2 = stats_2022['n_total']
    
    p_pooled = (stats_2021['n_education'] + stats_2022['n_education']) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed
    
    output_file = os.path.join(results_dir, 'education_shift_hypothesis_test.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Educational Focus Shift Hypothesis Test (2021 vs 2022) ===\n")
        f.write("\nPeriod Comparison: May-June 2021 vs May-June 2022\n")
        
        f.write("\nHypothesis Test Results:")
        f.write("\n" + "="*50)
        f.write("\nNull Hypothesis (H₀): The proportion of educational content views remained the same")
        f.write("\nAlternative Hypothesis (H₁): There was a significant change in the proportion of educational content views")
        f.write("\nTest Used: Two-proportion z-test")
        
        f.write("\n\n2021 Statistics:")
        f.write(f"\nTotal Views: {stats_2021['n_total']}")
        f.write(f"\nEducation Views: {stats_2021['n_education']}")
        f.write(f"\nEducation Proportion: {stats_2021['proportion']*100:.1f}%")
        f.write(f"\nTotal Hours: {stats_2021['total_hours']:.2f}")
        f.write(f"\nMean Duration: {stats_2021['formatted_mean']} (hh:mm)")
        f.write(f"\nMedian Duration: {stats_2021['formatted_median']} (hh:mm)")
        f.write("\n\nDuration Distribution:")
        for category, percentage in stats_2021['duration_distribution'].items():
            f.write(f"\n{category}: {percentage:.1f}%")
        
        f.write("\n\n2022 Statistics:")
        f.write(f"\nTotal Views: {stats_2022['n_total']}")
        f.write(f"\nEducation Views: {stats_2022['n_education']}")
        f.write(f"\nEducation Proportion: {stats_2022['proportion']*100:.1f}%")
        f.write(f"\nTotal Hours: {stats_2022['total_hours']:.2f}")
        f.write(f"\nMean Duration: {stats_2022['formatted_mean']} (hh:mm)")
        f.write(f"\nMedian Duration: {stats_2022['formatted_median']} (hh:mm)")
        f.write("\n\nDuration Distribution:")
        for category, percentage in stats_2022['duration_distribution'].items():
            f.write(f"\n{category}: {percentage:.1f}%")
        
        f.write("\n\nZ-Test Results:")
        f.write(f"\nz-statistic: {z_stat:.4f}")
        f.write(f"\np-value: {p_value:.6f}")
        
        alpha = 0.05
        f.write(f"\n\nAt α = {alpha}:")
        if p_value < alpha:
            f.write("\nReject H₀: There is strong statistical evidence of a change")
            f.write("\nin the proportion of educational content views between 2021 and 2022 (p < 0.05)")
        else:
            f.write("\nFail to reject H₀: There is not enough statistical evidence")
            f.write("\nof a change in educational content proportion (p ≥ 0.05)")
        
        proportion_change = ((stats_2022['proportion'] - stats_2021['proportion']) /
                             stats_2021['proportion'] * 100) if stats_2021['proportion'] != 0 else 0
        f.write(f"\n\nYear-over-Year Change: {proportion_change:.1f}%")
        f.write("\n\nNote: A positive change indicates an increase in educational content proportion")

def perform_market_correlation_analysis(data, results_dir):
    """
    Analyze correlation between stock market video watching duration and XU100 performance
    """
    # Market-related content
    market_data = data[
        ((data['category'] == 'Education') & (data['subcategory'] == 'finance_market')) |
        (data['category'] == 'Finance')
    ].copy()
    
    # Group by month
    monthly_viewing = market_data.groupby('month')['duration'].sum().reset_index()
    monthly_viewing = monthly_viewing.sort_values('month')
    
    # Approximate XU100 values for the relevant months
    xu100_data = pd.DataFrame({
        'month': ['2023-06', '2023-07', '2023-08', '2023-09'],
        'xu100': [5000, 5500, 6500, 7000]  # Example values
    })
    
    # Merge
    correlation_data = pd.merge(monthly_viewing, xu100_data, on='month', how='inner')
    
    # Pearson correlation
    correlation_coef = correlation_data['duration'].corr(correlation_data['xu100'])
    
    output_file = os.path.join(results_dir, 'market_correlation_analysis.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Stock Market Content and XU100 Correlation Analysis ===\n")
        f.write("\nPeriod Coverage: June 2023 - September 2023\n")
        
        f.write("\nMonthly Statistics:")
        f.write("\n" + "="*50)
        
        for _, row in correlation_data.iterrows():
            f.write(f"\n\n{row['month']}:")
            f.write(f"\nViewing Duration: {row['duration']:.2f} hours")
            f.write(f"\nXU100 Value: {row['xu100']}")
        
        f.write("\n\nCorrelation Analysis:")
        f.write("\n" + "="*50)
        f.write(f"\nPearson Correlation Coefficient: {correlation_coef:.4f}")
        
        # Interpretation
        if abs(correlation_coef) > 0.7:
            strength = "strong"
        elif abs(correlation_coef) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        direction = "positive" if correlation_coef > 0 else "negative"
        
        f.write(f"\nThere is a {strength} {direction} correlation between")
        f.write("\nstock market content viewing duration and XU100 performance.")
        
        # Monthly growth rates
        f.write("\n\nMonthly Growth Rates:")
        f.write("\n" + "="*50)
        
        for i in range(1, len(correlation_data)):
            prev_duration = correlation_data.iloc[i-1]['duration']
            curr_duration = correlation_data.iloc[i]['duration']
            duration_growth = ((curr_duration - prev_duration) / prev_duration) * 100 if prev_duration != 0 else 0
            
            prev_xu100 = correlation_data.iloc[i-1]['xu100']
            curr_xu100 = correlation_data.iloc[i]['xu100']
            xu100_growth = ((curr_xu100 - prev_xu100) / prev_xu100) * 100 if prev_xu100 != 0 else 0
            
            month = correlation_data.iloc[i]['month']
            f.write(f"\n\n{month}:")
            f.write(f"\nViewing Duration Growth: {duration_growth:.1f}%")
            f.write(f"\nXU100 Growth: {xu100_growth:.1f}%")

def hypothesis_tests_main():
    """
    Orchestrates the hypothesis tests on 'enhanced_categories.csv':
      1. Gaming peak period hypothesis test
      2. Educational shift between May-June 2021 and 2022
      3. Market correlation analysis
    """
    enhanced_df = pd.read_csv('data/enhanced_categories.csv')
    
    enhanced_df['watched_on'] = pd.to_datetime(enhanced_df['watched_on'], format='ISO8601')
    enhanced_df['month'] = enhanced_df['watched_on'].dt.strftime('%Y-%m')
    
    results_dir = 'hypothesis_test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Gaming Peak Hypothesis
    peak_months = [
        '2020-05','2020-06','2020-07','2020-08','2020-09',
        '2020-10','2020-11','2020-12','2021-01','2021-02'
    ]
    peak_data = enhanced_df[enhanced_df['month'].isin(peak_months)].copy()
    peak_data = peak_data[peak_data['duration'] <= 24]
    perform_gaming_hypothesis_test(peak_data, results_dir)
    
    # 2. Educational Shift
    may_june_2021 = enhanced_df[enhanced_df['month'].isin(['2021-05', '2021-06'])].copy()
    may_june_2022 = enhanced_df[enhanced_df['month'].isin(['2022-05', '2022-06'])].copy()
    
    may_june_2021 = may_june_2021[may_june_2021['duration'] <= 24]
    may_june_2022 = may_june_2022[may_june_2022['duration'] <= 24]
    
    perform_education_shift_hypothesis_test(may_june_2021, may_june_2022, results_dir)
    
    # 3. Market Correlation
    perform_market_correlation_analysis(enhanced_df, results_dir)
    
    print("Hypothesis tests completed. Results saved in 'hypothesis_test_results' directory")


###############################################################################
#                      SINGLE ENTRY POINT FOR EVERYTHING                      #
###############################################################################

if __name__ == "__main__":
    # 1. Run category analysis on temporal_patterns.csv
    category_analysis_main()
    
    # 2. Run channel/peak analysis on enhanced_categories.csv & channel_categories.csv
    analyze_all_peaks()
    
    # 3. Run all hypothesis tests on enhanced_categories.csv
    hypothesis_tests_main()

    print("\nAll analyses complete!")

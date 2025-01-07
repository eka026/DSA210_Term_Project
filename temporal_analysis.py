import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy.stats import kruskal, ttest_ind

def create_analysis_plots(df, output_dir='results'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare time-based columns, maintaining timezone consistency
    df['date'] = df['watched_on'].dt.date
    df['hour'] = df['watched_on'].dt.hour
    df['week'] = df['watched_on'].dt.isocalendar().week
    # Use strftime to extract month as 'YYYY-MM' to avoid dropping timezone info
    df['month'] = df['watched_on'].dt.strftime('%Y-%m')
    df['day_of_week'] = df['watched_on'].dt.day_name()
    
    # Set style for all plots
    sns.set_style("whitegrid")
    
    # 1. Monthly Watch Time Pattern
    plt.figure(figsize=(15, 8))
    monthly_watch = df.groupby('month')['duration'].sum()
    
    # Create the line plot
    ax = sns.lineplot(data=monthly_watch, marker='o', markersize=8)
    
    # Customize the plot
    plt.title('Monthly Watch Time Pattern', fontsize=16, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Hours Watched', fontsize=12)
    
    # Show only every nth label to avoid overcrowding
    n = 3  # Adjust this value to show more or fewer labels
    plt.xticks(range(0, len(monthly_watch), n), 
               [monthly_watch.index[i] for i in range(0, len(monthly_watch), n)],
               rotation=45, ha='right')
    
    # Add value labels with better positioning
    for idx, val in enumerate(monthly_watch.values):
        if idx % n == 0:  # Only label every nth point
            plt.text(idx, val + max(monthly_watch.values) * 0.02,  # Add small vertical offset
                    f'{val:.1f}h',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Category Distribution
    plt.figure(figsize=(15, 8))
    
    # Calculate total duration for each category
    category_totals = df.groupby('category')['duration'].sum().sort_values(ascending=False)
    
    # Identify major categories (top 5) and group others
    top_categories = category_totals.head(5).index.tolist()
    
    # Create a function to map categories
    def map_category(cat):
        return cat if cat in top_categories else 'Other'
    
    df['category_grouped'] = df['category'].map(map_category)
    
    # Create monthly aggregation
    monthly_category = df.pivot_table(
        values='duration',
        index='month',
        columns='category_grouped',
        aggfunc='sum',
        fill_value=0
    )
    
    # Sort by month
    monthly_category = monthly_category.sort_index()
    
    # Create stacked area chart
    ax = monthly_category.plot(
        kind='area',
        stacked=True,
        figsize=(15, 8),
        alpha=0.75  # Add some transparency
    )
    
    plt.title('Monthly Watch Time by Category', fontsize=16, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Hours Watched', fontsize=12)
    
    # Show every third month
    n = 3  # Show every 3rd month
    plt.xticks(range(0, len(monthly_category), n), 
               [monthly_category.index[i] for i in range(0, len(monthly_category), n)],
               rotation=45, ha='right')
    
    # Customize legend
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_category_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Hourly Distribution
    plt.figure(figsize=(15, 8))
    hourly_pattern = df.groupby('hour')['duration'].sum()
    sns.barplot(x=hourly_pattern.index, y=hourly_pattern.values)
    plt.title('Watch Time Distribution by Hour', fontsize=14, pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Total Hours Watched', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(hourly_pattern.values):
        plt.text(i, v, f'{v:.1f}h', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Day of Week Distribution
    plt.figure(figsize=(15, 8))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_pattern = df.groupby('day_of_week')['duration'].sum()
    daily_pattern = daily_pattern.reindex(day_order)
    sns.barplot(x=daily_pattern.index, y=daily_pattern.values)
    plt.title('Watch Time Distribution by Day of Week', fontsize=14, pad=20)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Total Hours Watched', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(daily_pattern.values):
        plt.text(i, v, f'{v:.1f}h', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Hypothesis Testing: Kruskal-Wallis H Test
    # Prepare data for Kruskal-Wallis H test
    groups = [group['duration'].values for name, group in df.groupby('month')]
    
    # Perform Kruskal-Wallis H test
    h_stat, p_value = kruskal(*groups)
    
    # Determine the conclusion based on p-value
    alpha = 0.05
    if p_value < alpha:
        conclusion = "Reject the null hypothesis: There are significant differences in viewing duration distributions across months."
    else:
        conclusion = "Fail to reject the null hypothesis: No significant differences in viewing duration distributions across months."
    
    # Save summary statistics to a text file
    with open(os.path.join(output_dir, 'viewing_statistics.txt'), 'w') as f:
        f.write("Viewing Pattern Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # Perform t-test for weekday vs weekend comparison
        daily_totals = df.groupby(['date', 'day_of_week'])['duration'].sum().reset_index()
        weekday_totals = daily_totals[daily_totals['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['duration']
        weekend_totals = daily_totals[daily_totals['day_of_week'].isin(['Saturday', 'Sunday'])]['duration']
        
        # Calculate statistics
        weekday_total = weekday_totals.sum()
        weekend_total = weekend_totals.sum()
        weekday_daily_avg = weekday_total / len(weekday_totals)
        weekend_daily_avg = weekend_total / len(weekend_totals)
        
        # Perform t-test on daily totals
        t_stat, p_value_t = ttest_ind(weekday_totals, weekend_totals)
        
        # Determine conclusion for t-test
        alpha = 0.05
        if p_value_t < alpha:
            weekday_weekend_conclusion = "Reject the null hypothesis: There is a significant difference in watch time between weekdays and weekends."
        else:
            weekday_weekend_conclusion = "Fail to reject the null hypothesis: There is no significant difference in watch time between weekdays and weekends."

        f.write("Weekday vs Weekend Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total weekday watch time: {weekday_total:.1f} hours\n")
        f.write(f"Total weekend watch time: {weekend_total:.1f} hours\n")
        f.write(f"Average daily weekday watch time: {weekday_daily_avg:.1f} hours per day\n")
        f.write(f"Average daily weekend watch time: {weekend_daily_avg:.1f} hours per day\n")
        f.write(f"\nStatistical Test Results:\n")
        f.write(f"T-statistic: {t_stat:.4f}\n")
        if p_value_t < 0.001:
            f.write("P-value: p < 0.001\n")
        else:
            f.write(f"P-value: {p_value_t:.4f}\n")
        f.write(f"Conclusion: {weekday_weekend_conclusion}\n\n")
        
        f.write("General Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total watch time: {df['duration'].sum():.1f} hours\n")
        f.write(f"Average monthly watch time: {monthly_watch.mean():.1f} hours\n")
        f.write(f"Peak hour: {hourly_pattern.idxmax()}:00 ({hourly_pattern.max():.1f} hours total)\n")
        f.write(f"Most active day: {daily_pattern.idxmax()} ({daily_pattern.max():.1f} hours total)\n\n")
        
        f.write("Monthly Statistics:\n")
        f.write("-" * 40 + "\n")
        for month, duration in monthly_watch.items():
            f.write(f"{month}: {duration:.1f} hours\n")
        
        f.write("\nHypothesis Testing: Temporal Analysis\n")
        f.write("-" * 40 + "\n")
        f.write("Null Hypothesis: The distributions of video watching duration are the same across all months.\n")
        f.write("Alternative Hypothesis: The distributions of video watching duration differ across months.\n\n")
        f.write("Note: Using Kruskal-Wallis H test instead of ANOVA because:\n")
        f.write("1. The data is not normally distributed\n")
        f.write("2. There are significant outliers in the viewing durations\n")
        f.write("3. The test makes no assumptions about the distribution of the data\n\n")
        f.write(f"Kruskal-Wallis H Test Results:\n")
        f.write(f"H-statistic: {h_stat:.4f}\n")
        # Format p-value with a conditional check
        if p_value < 0.001:
            f.write(f"P-value: p < 0.001\n")
        else:
            f.write(f"P-value: {p_value:.4f}\n")  # Format p-value in scientific notation
        f.write(f"Conclusion: {conclusion}\n")

        f.write(f"Conclusion: {conclusion}\n\n")
        
        # Time Period Analysis (Morning vs Afternoon vs Evening)
        f.write("\nTime Period Analysis\n")
        f.write("-" * 40 + "\n")
        f.write("Comparing watch time distributions across different periods of the day:\n")
        f.write("- Morning: 06:00-11:59\n")
        f.write("- Afternoon: 12:00-17:59\n")
        f.write("- Evening: 18:00-23:59\n")
        f.write("- Night: 00:00-05:59 (excluded from comparison)\n\n")
        
        # Define time periods (24-hour format)
        def get_time_period(hour):
            if 6 <= hour < 12:  # 06:00-11:59
                return 'morning'
            elif 12 <= hour < 18:  # 12:00-17:59
                return 'afternoon'
            elif 18 <= hour < 24:  # 18:00-23:59
                return 'evening'
            else:  # 00:00-05:59
                return 'night'
        
        df['time_period'] = df['hour'].apply(get_time_period)
        
        # Get data for each period
        morning_data = df[df['time_period'] == 'morning']['duration']
        afternoon_data = df[df['time_period'] == 'afternoon']['duration']
        evening_data = df[df['time_period'] == 'evening']['duration']
        
        # Perform Kruskal-Wallis H test for time periods
        h_stat_time, p_value_time = kruskal(morning_data, afternoon_data, evening_data)
        
        # Calculate total watch time for each period
        period_totals = df[df['time_period'].isin(['morning', 'afternoon', 'evening'])].groupby('time_period')['duration'].sum()
        
        f.write("Total Watch Time by Period:\n")
        for period, total in period_totals.items():
            f.write(f"{period.capitalize()}: {total:.1f} hours\n")
            
        # Calculate hourly averages (total for period divided by number of hours in period)
        hours_per_period = {'morning': 6, 'afternoon': 6, 'evening': 6}  # each period is 6 hours
        f.write("\nAverage Watch Time per Hour by Period:\n")
        for period in period_totals.index:
            avg_per_hour = period_totals[period] / hours_per_period[period]
            f.write(f"{period.capitalize()}: {avg_per_hour:.1f} hours per hour\n")
        
        f.write("\nKruskal-Wallis H Test Results:\n")
        f.write(f"H-statistic: {h_stat_time:.4f}\n")
        if p_value_time < 0.001:
            f.write("P-value: p < 0.001\n")
        else:
            f.write(f"P-value: {p_value_time:.4f}\n")
            
        # Determine conclusion for time period analysis
        if p_value_time < 0.05:
            time_period_conclusion = "Reject the null hypothesis: There are significant differences in viewing duration distributions across time periods."
        else:
            time_period_conclusion = "Fail to reject the null hypothesis: No significant differences in viewing duration distributions across time periods."
            
        f.write(f"Conclusion: {time_period_conclusion}\n")
    
    return {
        'monthly_watch': monthly_watch,
        'hourly_pattern': hourly_pattern,
        'daily_pattern': daily_pattern,
        'kruskal_wallis': {
            'h_statistic': h_stat,
            'p_value': p_value,
            'conclusion': conclusion
        }
    }

def analyze_viewing_patterns(enhanced_categories_path, output_dir='results'):
    # Read the data
    df = pd.read_csv(enhanced_categories_path)
    
    # Fix datetime parsing using ISO format and set timezone
    # Removed any 'format' parameter to let pandas infer the format
    df['watched_on'] = pd.to_datetime(df['watched_on'], utc=True, errors='coerce')
    
    # Check for any parsing errors
    num_invalid_dates = df['watched_on'].isna().sum()
    if num_invalid_dates > 0:
        print(f"Warning: {num_invalid_dates} invalid datetime entries were found and set to NaT.")
        # Optionally, you can drop these rows or handle them as needed
        df = df.dropna(subset=['watched_on'])
        print(f"Rows with invalid datetime entries have been removed. Remaining data points: {len(df)}")
    
    # Create the analysis plots and get the statistics
    stats = create_analysis_plots(df, output_dir)
    
    print(f"\nAnalysis complete! Files have been saved to the '{output_dir}' directory:")
    print("1. monthly_pattern.png - Monthly watching patterns")
    print("2. monthly_category_pattern.png - Monthly patterns by category")
    print("3. hourly_pattern.png - Distribution of watch time by hour")
    print("4. daily_distribution.png - Distribution of watch time by day of week")
    print("5. viewing_statistics.txt - Detailed statistics and patterns, including ANOVA results")

if __name__ == "__main__":
    # Specify the paths
    enhanced_categories_path = "data/enhanced_categories.csv"
    output_dir = "results"
    
    # Run the analysis
    analyze_viewing_patterns(enhanced_categories_path, output_dir)
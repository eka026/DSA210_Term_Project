import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def create_analysis_plots(df, output_dir='results'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare time-based columns, maintaining timezone consistency
    df['date'] = df['watched_on'].dt.date
    df['hour'] = df['watched_on'].dt.hour
    df['week'] = df['watched_on'].dt.isocalendar().week
    # Use month start date instead of period for better timezone handling
    df['month'] = df['watched_on'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['watched_on'].dt.day_name()
    
    # Set style for all plots
    sns.set_style("whitegrid")
    
    # 1. Monthly Watch Time Pattern
    plt.figure(figsize=(15, 8))
    monthly_watch = df.groupby('month')['duration'].sum()
    sns.lineplot(data=monthly_watch)
    
    plt.title('Monthly Watch Time Pattern', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Hours Watched', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on points
    for idx, val in enumerate(monthly_watch.values):
        plt.text(idx, val, f'{val:.1f}h', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Daily Watch Time Pattern
    plt.figure(figsize=(15, 8))
    daily_watch = df.groupby('date')['duration'].sum()
    sns.lineplot(data=daily_watch)
    
    # Calculate and annotate significant spikes and drops
    mean_watch = daily_watch.mean()
    std_watch = daily_watch.std()
    spikes = daily_watch[daily_watch > mean_watch + 2*std_watch]
    drops = daily_watch[daily_watch < mean_watch - 2*std_watch]
    
    for idx, val in spikes.items():
        plt.annotate(f'{val:.1f}h', 
                    xy=(idx, val), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    color='green')
    
    for idx, val in drops.items():
        plt.annotate(f'{val:.1f}h', 
                    xy=(idx, val), 
                    xytext=(10, -10),
                    textcoords='offset points',
                    color='red')
    
    plt.title('Daily Watch Time Pattern', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Hours Watched', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Monthly Category Distribution
    plt.figure(figsize=(15, 8))
    monthly_category = df.pivot_table(
        values='duration',
        index='month',
        columns='category',
        aggfunc='sum',
        fill_value=0
    )
    monthly_category.plot(kind='bar', stacked=True)
    plt.title('Monthly Watch Time by Category', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Hours Watched', fontsize=12)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_category_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Hourly Distribution
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
    
    # 5. Day of Week Distribution
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
    
    # Save summary statistics to a text file
    with open(os.path.join(output_dir, 'viewing_statistics.txt'), 'w') as f:
        f.write("Viewing Pattern Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("General Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total watch time: {df['duration'].sum():.1f} hours\n")
        f.write(f"Average daily watch time: {daily_watch.mean():.1f} hours\n")
        f.write(f"Average monthly watch time: {monthly_watch.mean():.1f} hours\n")
        f.write(f"Peak hour: {hourly_pattern.idxmax()}:00 ({hourly_pattern.max():.1f} hours total)\n")
        f.write(f"Most active day: {daily_pattern.idxmax()} ({daily_pattern.max():.1f} hours total)\n\n")
        
        f.write("Monthly Statistics:\n")
        f.write("-" * 40 + "\n")
        for month, duration in monthly_watch.items():
            f.write(f"{month}: {duration:.1f} hours\n")
        
        f.write("\nSignificant Patterns:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of high-activity days (>2 std): {len(spikes)}\n")
        f.write(f"Number of low-activity days (<2 std): {len(drops)}\n\n")
        
        f.write("Peak watching days:\n")
        f.write("-" * 40 + "\n")
        for date, hours in spikes.nlargest(5).items():
            f.write(f"- {date}: {hours:.1f} hours\n")
    
    return {
        'monthly_watch': monthly_watch,
        'daily_watch': daily_watch,
        'hourly_pattern': hourly_pattern,
        'daily_pattern': daily_pattern,
        'spikes': spikes,
        'drops': drops
    }

def analyze_viewing_patterns(enhanced_categories_path, output_dir='results'):
    # Read the data
    df = pd.read_csv(enhanced_categories_path)
    
    # Fix datetime parsing using ISO format and set timezone
    df['watched_on'] = pd.to_datetime(df['watched_on'], format='ISO8601', utc=True)
    
    # Create the analysis plots and get the statistics
    stats = create_analysis_plots(df, output_dir)
    
    print(f"\nAnalysis complete! Files have been saved to the '{output_dir}' directory:")
    print("1. monthly_pattern.png - Monthly watching patterns")
    print("2. daily_pattern.png - Daily watching patterns with spikes and drops")
    print("3. monthly_category_pattern.png - Monthly patterns by category")
    print("4. hourly_pattern.png - Distribution of watch time by hour")
    print("5. daily_distribution.png - Distribution of watch time by day of week")
    print("6. viewing_statistics.txt - Detailed statistics and patterns")

if __name__ == "__main__":
    # Specify the paths
    enhanced_categories_path = "data/enhanced_categories.csv"
    output_dir = "results"
    
    # Run the analysis
    analyze_viewing_patterns(enhanced_categories_path, output_dir)
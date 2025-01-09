import pandas as pd
import numpy as np
from scipy import stats
import os

def perform_gaming_hypothesis_test(data, results_dir):
    """
    Perform hypothesis test comparing gaming vs non-gaming watch time
    for the Gaming Peak period (2020-2021)
    """
    # Split data into gaming and non-gaming groups
    gaming_data = data[data['category'] == 'Gaming']['duration']
    nongaming_data = data[data['category'] != 'Gaming']['duration']
    
    # Calculate descriptive statistics
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
    
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(
        gaming_data, 
        nongaming_data,
        alternative='greater'  # One-tailed test: gaming > non-gaming
    )
    
    # Write results to a new file
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
        
        # Add interpretation
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
                "15-30min": len(durations[(durations > 0.25) & (durations <= 0.5)]) / len(durations) * 100,
                "30-60min": len(durations[(durations > 0.5) & (durations <= 1.0)]) / len(durations) * 100,
                "> 1hour": len(durations[durations > 1.0]) / len(durations) * 100
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
    
    # Calculate z-test statistic
    p1 = stats_2021['proportion']
    p2 = stats_2022['proportion']
    n1 = stats_2021['n_total']
    n2 = stats_2022['n_total']
    
    # Pooled proportion
    p_pooled = (stats_2021['n_education'] + stats_2022['n_education']) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Z-statistic
    z_stat = (p2 - p1) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Write results to a new file
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
        
        # Add interpretation
        alpha = 0.05
        f.write(f"\n\nAt α = {alpha}:")
        if p_value < alpha:
            f.write("\nReject H₀: There is strong statistical evidence of a change")
            f.write("\nin the proportion of educational content views between 2021 and 2022 (p < 0.05)")
        else:
            f.write("\nFail to reject H₀: There is not enough statistical evidence")
            f.write("\nof a change in educational content proportion (p ≥ 0.05)")
        
        # Add effect size
        proportion_change = ((stats_2022['proportion'] - 
                            stats_2021['proportion']) / 
                           stats_2021['proportion'] * 100)
        f.write(f"\n\nYear-over-Year Change: {proportion_change:.1f}%")
        f.write("\n\nNote: A positive change indicates an increase in educational content proportion")

def perform_market_correlation_analysis(data, results_dir):
    """
    Analyze correlation between stock market video watching duration and XU100 performance
    """
    # Filter for market-related content
    market_data = data[
        ((data['category'] == 'Education') & (data['subcategory'] == 'finance_market')) |
        (data['category'] == 'Finance')
    ].copy()
    
    # Group by month and calculate total duration
    monthly_viewing = market_data.groupby('month')['duration'].sum().reset_index()
    monthly_viewing = monthly_viewing.sort_values('month')
    
    # Define XU100 values for the relevant months (approximate values)
    xu100_data = pd.DataFrame({
        'month': ['2023-06', '2023-07', '2023-08', '2023-09'],
        'xu100': [5000, 5500, 6500, 7000]  # Approximate values
    })
    
    # Merge viewing data with XU100 data
    correlation_data = pd.merge(monthly_viewing, xu100_data, on='month', how='inner')
    
    # Calculate Pearson correlation
    correlation_coef = correlation_data['duration'].corr(correlation_data['xu100'])
    
    # Write results to a new file
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
        
        # Add interpretation
        f.write("\n\nInterpretation:")
        if abs(correlation_coef) > 0.7:
            strength = "strong"
        elif abs(correlation_coef) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
            
        direction = "positive" if correlation_coef > 0 else "negative"
        
        f.write(f"\nThere is a {strength} {direction} correlation between")
        f.write("\nstock market content viewing duration and XU100 performance.")
        
        # Add monthly growth rates
        f.write("\n\nMonthly Growth Rates:")
        f.write("\n" + "="*50)
        
        # Calculate viewing duration growth
        for i in range(1, len(correlation_data)):
            prev_duration = correlation_data.iloc[i-1]['duration']
            curr_duration = correlation_data.iloc[i]['duration']
            duration_growth = ((curr_duration - prev_duration) / prev_duration) * 100
            
            prev_xu100 = correlation_data.iloc[i-1]['xu100']
            curr_xu100 = correlation_data.iloc[i]['xu100']
            xu100_growth = ((curr_xu100 - prev_xu100) / prev_xu100) * 100
            
            month = correlation_data.iloc[i]['month']
            f.write(f"\n\n{month}:")
            f.write(f"\nViewing Duration Growth: {duration_growth:.1f}%")
            f.write(f"\nXU100 Growth: {xu100_growth:.1f}%")

def analyze_data():
    # Read and prepare data
    enhanced_df = pd.read_csv('data/enhanced_categories.csv')
    
    # Basic data cleaning - using ISO8601 format
    enhanced_df['watched_on'] = pd.to_datetime(enhanced_df['watched_on'], format='ISO8601')
    enhanced_df['month'] = enhanced_df['watched_on'].dt.strftime('%Y-%m')
    
    # Create results directory
    results_dir = 'hypothesis_test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Gaming Peak Analysis
    peak_months = ['2020-05', '2020-06', '2020-07', '2020-08', '2020-09', 
                  '2020-10', '2020-11', '2020-12', '2021-01', '2021-02']
    peak_data = enhanced_df[enhanced_df['month'].isin(peak_months)].copy()
    peak_data = peak_data[peak_data['duration'] <= 24]  # Remove outliers
    perform_gaming_hypothesis_test(peak_data, results_dir)
    
    # 2. Educational Shift Analysis
    may_june_2021 = enhanced_df[enhanced_df['month'].isin(['2021-05', '2021-06'])].copy()
    may_june_2022 = enhanced_df[enhanced_df['month'].isin(['2022-05', '2022-06'])].copy()
    
    # Remove outliers
    may_june_2021 = may_june_2021[may_june_2021['duration'] <= 24]
    may_june_2022 = may_june_2022[may_june_2022['duration'] <= 24]
    
    perform_education_shift_hypothesis_test(may_june_2021, may_june_2022, results_dir)

    perform_market_correlation_analysis(enhanced_df, results_dir)
    
    print("Hypothesis tests completed. Results saved in 'hypothesis_test_results' directory")

if __name__ == "__main__":
    analyze_data()
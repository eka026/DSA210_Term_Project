import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Read the enhanced categories data
df = pd.read_csv('data/enhanced_categories.csv')

# Convert watched_on to datetime using ISO format
df['watched_on'] = pd.to_datetime(df['watched_on'], format='ISO8601')

# Create different time aggregations
daily_counts = df.groupby(df['watched_on'].dt.date).size().reset_index()
daily_counts.columns = ['date', 'count']

weekly_counts = df.groupby(pd.Grouper(key='watched_on', freq='W')).size().reset_index()
weekly_counts.columns = ['date', 'count']

# Changed from 'M' to 'ME' for month-end frequency
monthly_counts = df.groupby(pd.Grouper(key='watched_on', freq='ME')).size().reset_index()
monthly_counts.columns = ['date', 'count']

# Create daily view count plot
daily_fig = px.line(daily_counts, x='date', y='count',
                    title='Daily Watch Count',
                    labels={'date': 'Date', 'count': 'Number of Videos Watched'},
                    template='plotly_white')

daily_fig.update_layout(
    showlegend=False,
    xaxis_title="Date",
    yaxis_title="Number of Videos",
    hovermode='x unified',
    width=1200,
    height=600
)

# Add markers for spikes (points above mean + 1.5*std)
mean_count = daily_counts['count'].mean()
std_count = daily_counts['count'].std()
spike_threshold = mean_count + 1.5 * std_count

spikes = daily_counts[daily_counts['count'] > spike_threshold]

daily_fig.add_trace(
    go.Scatter(
        x=spikes['date'],
        y=spikes['count'],
        mode='markers',
        name='Spikes',
        marker=dict(color='red', size=8),
        hovertemplate="Date: %{x}<br>Count: %{y}<br>Spike in activity<extra></extra>"
    )
)

# Create weekly view count plot
weekly_fig = px.line(weekly_counts, x='date', y='count',
                     title='Weekly Watch Count',
                     labels={'date': 'Week', 'count': 'Number of Videos Watched'},
                     template='plotly_white')

weekly_fig.update_layout(
    showlegend=False,
    xaxis_title="Week",
    yaxis_title="Number of Videos",
    hovermode='x unified',
    width=1200,
    height=600
)

# Create monthly view count plot
monthly_fig = px.line(monthly_counts, x='date', y='count',
                      title='Monthly Watch Count',
                      labels={'date': 'Month', 'count': 'Number of Videos Watched'},
                      template='plotly_white')

monthly_fig.update_layout(
    showlegend=False,
    xaxis_title="Month",
    yaxis_title="Number of Videos",
    hovermode='x unified',
    width=1200,
    height=600
)

# Ensure the visualizations directory exists
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

try:
    # Save the plots as PNG files with better resolution
    daily_fig.write_image("visualizations/daily_watch_count.png", scale=2)
    weekly_fig.write_image("visualizations/weekly_watch_count.png", scale=2)
    monthly_fig.write_image("visualizations/monthly_watch_count.png", scale=2)
    print("\nVisualization files have been saved in the 'visualizations' directory.")
except Exception as e:
    print(f"\nError saving visualization files: {str(e)}")
    print("Please make sure you have installed kaleido using: pip install -U kaleido")
    # Save as HTML as fallback
    print("Saving as HTML files instead...")
    daily_fig.write_html("visualizations/daily_watch_count.html")
    weekly_fig.write_html("visualizations/weekly_watch_count.html")
    monthly_fig.write_html("visualizations/monthly_watch_count.html")
    print("HTML files have been saved in the 'visualizations' directory.")

# Print some basic statistics
print("\nBasic Statistics:")
print("\nDaily Statistics:")
print(f"Average daily watches: {mean_count:.2f}")
print(f"Standard deviation: {std_count:.2f}")
print(f"Number of spike days: {len(spikes)}")
print("\nTop 5 days with highest activity:")
print(daily_counts.nlargest(5, 'count')[['date', 'count']])

print("\nWeekly Statistics:")
print(f"Average weekly watches: {weekly_counts['count'].mean():.2f}")
print(f"Standard deviation: {weekly_counts['count'].std():.2f}")
print("\nTop 3 weeks with highest activity:")
print(weekly_counts.nlargest(3, 'count')[['date', 'count']])

print("\nMonthly Statistics:")
print(f"Average monthly watches: {monthly_counts['count'].mean():.2f}")
print(f"Standard deviation: {monthly_counts['count'].std():.2f}")
print("\nTop 3 months with highest activity:")
print(monthly_counts.nlargest(3, 'count')[['date', 'count']])
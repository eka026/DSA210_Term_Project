import json
import csv
from collections import defaultdict
from datetime import datetime
import re

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object, handling different formats."""
    try:
        # Try parsing with milliseconds
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            # Try parsing without milliseconds
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # If both fail, print the problematic timestamp and return None
            print(f"Warning: Could not parse timestamp: {timestamp_str}")
            return None

def extract_subcategories(title, description, category):
    """Extract subcategories based on video content."""
    # Combine title and description for analysis
    text = f"{title} {description}".lower()
    
    # Category-specific keywords
    category_patterns = {
        'Education': {
            'finance_market': r'(bist|borsa|xu100|thyao|garan|teknik analiz|hisse)',
            'financial_education': r'(yatırım|finansal|ekonomi|portföy|temettü)',
            'exam_prep': r'(yks|tyt|ayt|sınav|test|üniversite|matematik)',
            'programming': r'(python|javascript|coding|programming|developer)',
            'statistics': r'(statistics|neyman|pearson|calculus|probability)',
            'language': r'(english|spanish|language|grammar|speaking)'
        },
        'Gaming': {
            'strategy': r'(hearts of iron|hoi4|strategy|civilization|paradox)',
            'minecraft': r'minecraft',
            'fps': r'(valorant|csgo|fps|shooter)',
            'gta': r'(gta|grand theft auto)',
            'streaming': r'(twitch|stream|clips|moments)'
        },
        'Entertainment': {
            'comedy': r'(funny|comedy|meme|jokes|laugh)',
            'reactions': r'(reaction|reacts|watching)',
            'shorts': r'#shorts',
            'highlights': r'(highlights|moments|best|clips)',
            'analysis': r'(explained|analysis|review|breakdown)'
        },
        'People & Blogs': {
            'sports_commentary': r'(spor|futbol|maç|yorum)',
            'education': r'(calculus|math|ders|eğitim)',
            'politics': r'(political|politics|news|kemal)',
            'lifestyle': r'(lifestyle|daily|vlog)',
            'commentary': r'(hasanabi|porçay|commentary|react)'
        }
    }
    
    if category in category_patterns:
        for subcategory, pattern in category_patterns[category].items():
            if re.search(pattern, text):
                return subcategory
    
    return 'other'

def analyze_watch_patterns():
    # Load watch history
    with open('watch-history.json', 'r', encoding='utf-8') as json_file:
        watch_history = json.load(json_file)
    
    # Create a dictionary to store watch data with timestamps
    watch_data = {}
    for entry in watch_history:
        if "titleUrl" in entry:
            video_id = entry["titleUrl"].split("v=")[-1]
            watch_time = entry["time"]
            watch_data[video_id] = {
                'timestamp': watch_time,
                'channel': entry.get('subtitles', [{}])[0].get('name', 'Unknown Channel')
            }
    
    # Load and process video details
    video_categories = defaultdict(lambda: defaultdict(int))
    channel_categories = defaultdict(lambda: defaultdict(int))
    temporal_patterns = defaultdict(lambda: defaultdict(int))
    
    with open('video_details.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        output_data = []
        
        for row in reader:
            video_id = row['video_id']
            if video_id in watch_data:
                category = row['category_name']
                subcategory = extract_subcategories(
                    row['title'],
                    row['description'],
                    category
                )
                
                # Get timestamp and parse it
                watch_time = watch_data[video_id]['timestamp']
                timestamp = parse_timestamp(watch_time)
                
                if timestamp:  # Only process if timestamp parsing was successful
                    month_key = timestamp.strftime('%Y-%m')
                    
                    # Update various statistics
                    video_categories[category][subcategory] += 1
                    channel_categories[watch_data[video_id]['channel']][category] += 1
                    temporal_patterns[month_key][category] += 1
                    
                    # Prepare output data
                    output_data.append({
                        'video_id': video_id,
                        'category': category,
                        'subcategory': subcategory,
                        'watched_on': watch_time,
                        'channel': watch_data[video_id]['channel']
                    })
    
    # Save enhanced categorization results
    with open('enhanced_categories.csv', 'w', encoding='utf-8', newline='') as output_file:
        fieldnames = ['video_id', 'category', 'subcategory', 'watched_on', 'channel']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    # Save temporal analysis
    with open('temporal_patterns.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['month', 'category', 'count'])
        for month in sorted(temporal_patterns.keys()):
            for category, count in temporal_patterns[month].items():
                writer.writerow([month, category, count])
    
    # Save channel analysis
    with open('channel_categories.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'category', 'count'])
        for channel in channel_categories:
            for category, count in channel_categories[channel].items():
                writer.writerow([channel, category, count])
    
    return {
        'video_categories': dict(video_categories),
        'temporal_patterns': dict(temporal_patterns),
        'channel_categories': dict(channel_categories)
    }

if __name__ == "__main__":
    results = analyze_watch_patterns()
    print("\nAnalysis complete! Files generated:")
    print("1. enhanced_categories.csv - Detailed video categorization")
    print("2. temporal_patterns.csv - Viewing patterns over time")
    print("3. channel_categories.csv - Channel-based analysis")
import json
import csv
from collections import defaultdict
from datetime import datetime
import re
import os

# Define data directory path
DATA_DIR = "data"
OUTPUT_DIR = "data"  # Output to the same data directory

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
            print(f"Warning: Could not parse timestamp: {timestamp_str}")
            return None

def extract_subcategories(title, description, category):
    """Extract subcategories based on video content."""
    text = f"{title} {description}".lower()
    
    category_patterns = {
        'Education': {
            'finance_market': r'(bist|borsa|xu100|thyao|garan|teknik analiz|hisse|stock|market|trading|forex)',
            'financial_education': r'(yatırım|finansal|ekonomi|portföy|temettü|investment|portfolio|dividend)',
            'exam_prep': r'(yks|tyt|ayt|sınav|test|üniversite|exam|sat|gre|gmat)',
            'programming': r'(python|javascript|java|coding|programming|developer|code|software)',
            'statistics': r'(statistics|neyman|pearson|calculus|probability|regression|machine learning)',
            'language': r'(english|spanish|language|grammar|speaking|writing|vocabulary)',
            'other_education': r'(education|learning|tutorial|course|lecture)'
        },
        'Gaming': {
            'strategy': r'(hearts of iron|hoi4|strategy|civilization|paradox|europa|crusader kings)',
            'minecraft': r'minecraft',
            'fps': r'(valorant|csgo|fps|shooter|call of duty|battlefield|overwatch)',
            'gta': r'(gta|grand theft auto)',
            'streaming': r'(twitch|stream|clips|moments|highlights)',
            'other_gaming': r''  # Default subcategory for Gaming
        },
        'Entertainment': {
            'comedy': r'(funny|comedy|meme|jokes|laugh|humor)',
            'reactions': r'(reaction|reacts|watching|reacting)',
            'shorts': r'#shorts',
            'highlights': r'(highlights|moments|best|clips|compilation)',
            'analysis': r'(explained|analysis|review|breakdown|theory)',
            'other_entertainment': r''  # Default subcategory for Entertainment
        },
        'People & Blogs': {
            'personal_vlogs': r'(vlog|daily|life|routine|lifestyle)',
            'commentary': r'(commentary|thoughts|opinion|discussing)',
            'tech_reviews': r'(review|tech|technology|gadget|phone|laptop)',
            'educational': r'(educational|learning|tutorial|how to)',
            'other_blogs': r''  # Default subcategory for People & Blogs
        }
    }
    
    if category in category_patterns:
        for subcategory, pattern in category_patterns[category].items():
            if pattern and re.search(pattern, text):
                return subcategory
        # If no specific pattern matches, return the default "other_" subcategory
        return f"other_{category.lower().replace(' & ', '_')}"
    
    return 'other'

def analyze_watch_patterns():
    # Load watch history from data directory
    watch_history_path = os.path.join(DATA_DIR, 'watch-history.json')
    with open(watch_history_path, 'r', encoding='utf-8') as json_file:
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
    
    # Initialize data structures
    video_categories = defaultdict(lambda: defaultdict(int))
    channel_categories = defaultdict(lambda: defaultdict(float))
    temporal_patterns = defaultdict(float)  # Will store (month, category, subcategory) tuples
    
    # Load and process video details from data directory
    video_details_path = os.path.join(DATA_DIR, 'video_details.csv')
    with open(video_details_path, 'r', encoding='utf-8') as csv_file:
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
                
                watch_time = watch_data[video_id]['timestamp']
                timestamp = parse_timestamp(watch_time)
                
                if timestamp:
                    month_key = timestamp.strftime('%Y-%m')
                    
                    try:
                        duration = float(row['duration'])
                    except (ValueError, TypeError):
                        duration = 0.0
                    
                    # Update statistics
                    video_categories[category][subcategory] += 1
                    channel_categories[watch_data[video_id]['channel']][category] += duration
                    temporal_patterns[(month_key, category, subcategory)] += duration
                    
                    output_data.append({
                        'video_id': video_id,
                        'category': category,
                        'subcategory': subcategory,
                        'watched_on': watch_time,
                        'channel': watch_data[video_id]['channel'],
                        'duration': duration
                    })
    
    # Save enhanced categorization results
    enhanced_categories_path = os.path.join(OUTPUT_DIR, 'enhanced_categories.csv')
    with open(enhanced_categories_path, 'w', encoding='utf-8', newline='') as output_file:
        fieldnames = ['video_id', 'category', 'subcategory', 'watched_on', 'channel', 'duration']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    # Save temporal analysis with subcategories
    temporal_patterns_path = os.path.join(OUTPUT_DIR, 'temporal_patterns.csv')
    with open(temporal_patterns_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['month', 'category', 'subcategory', 'duration'])
        # Sort by month, category, and subcategory
        sorted_patterns = sorted(temporal_patterns.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
        for (month, category, subcategory), duration in sorted_patterns:
            writer.writerow([month, category, subcategory, duration])
    
    # Save channel analysis
    channel_categories_path = os.path.join(OUTPUT_DIR, 'channel_categories.csv')
    with open(channel_categories_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'category', 'duration'])
        for channel in channel_categories:
            for category, duration in channel_categories[channel].items():
                writer.writerow([channel, category, duration])
    
    return {
        'video_categories': dict(video_categories),
        'temporal_patterns': dict(temporal_patterns),
        'channel_categories': dict(channel_categories)
    }

if __name__ == "__main__":
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        exit(1)
        
    # Check if required input files exist
    required_files = ['watch-history.json', 'video_details.csv']
    for file in required_files:
        file_path = os.path.join(DATA_DIR, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file '{file}' not found in {DATA_DIR}!")
            exit(1)
    
    print("Starting analysis...")
    results = analyze_watch_patterns()
    print("\nAnalysis complete! Files generated in", OUTPUT_DIR)
    print("1. enhanced_categories.csv - Detailed video categorization with durations")
    print("2. temporal_patterns.csv - Viewing patterns over time with durations (including subcategories)")
    print("3. channel_categories.csv - Channel-based analysis with durations")
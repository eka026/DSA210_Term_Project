import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    for resource in ['punkt', 'stopwords', 'punkt_tab']:
        nltk.download(resource, quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {str(e)}")
    print("Please try manually downloading the resources using:")
    print("import nltk")
    print("nltk.download('punkt')")
    print("nltk.download('stopwords')")
    print("nltk.download('punkt_tab')")

def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    # Find all hashtags in the text
    hashtags = re.findall(r'#(\w+)', str(text))
    # Clean each hashtag
    return [tag.lower() for tag in hashtags]

def clean_hashtag(tag):
    """Clean individual hashtag"""
    # List of words to exclude
    exclude_words = {
        'shorts', 'clips', 'kesfet', 'reklam', 'isbirligi', 
        'short', 'clip', 'kesif', 'sponsored', 'sponsorlu', 
        'sponsored_content', 'reel', 'reels', 'fyp'
    }
    
    # Remove special characters and digits, keep only letters
    cleaned = re.sub(r'[^\w\s]', '', tag.lower())
    cleaned = re.sub(r'\d+', '', cleaned)
    cleaned = cleaned.strip()
    
    # Return None if the cleaned tag is in exclude_words
    return None if cleaned in exclude_words else cleaned

def generate_hashtag_wordcloud(csv_path, min_word_length=3, max_words=100):
    """Generate and save word cloud from hashtags found in titles and descriptions"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
            
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {str(e)}")
        return
        
    # Extract and clean all hashtags from both title and description
    all_hashtags = []
    
    # Process titles
    title_hashtags = df['title'].apply(extract_hashtags)
    all_hashtags.extend([tag for tags in title_hashtags for tag in tags])
    
    # Process descriptions
    desc_hashtags = df['description'].apply(extract_hashtags)
    all_hashtags.extend([tag for tags in desc_hashtags for tag in tags])
    
    # Clean hashtags and filter out None values (excluded words)
    clean_hashtags = [cleaned for tag in all_hashtags 
                     if (cleaned := clean_hashtag(tag)) is not None 
                     and len(cleaned) >= min_word_length]
    
    if not clean_hashtags:
        print("No valid hashtags found in titles or descriptions")
        return
        
    # Join hashtags for word cloud
    hashtag_text = ' '.join(clean_hashtags)
    
    # Custom color function with colors that work well on white background
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = [
            (25, 25, 112),    # Midnight Blue
            (165, 42, 42),    # Brown
            (34, 139, 34),    # Forest Green
            (139, 0, 0),      # Dark Red
            (72, 61, 139),    # Dark Slate Blue
            (184, 134, 11),   # Dark Goldenrod
            (85, 107, 47),    # Dark Olive Green
            (128, 0, 128)     # Purple
        ]
        return colors[np.random.randint(0, len(colors))]
    
    # Create and generate word cloud with sparser layout and no duplicates
    wordcloud = WordCloud(
        width=1920,
        height=1080,
        background_color='white',
        color_func=color_func,
        min_word_length=3,
        max_words=60,              # Reduced number of words
        collocations=False,        # Disable collocations to prevent duplicates
        prefer_horizontal=0.75,
        relative_scaling=0.6,
        min_font_size=12,
        max_font_size=220,
        random_state=42,
        margin=8,                 # Increased margin for more spacing
        contour_width=0,
        contour_color='white'
    ).generate(hashtag_text)
    
    # Create figure with white background
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the word cloud
    output_path = os.path.join('results', 'hashtag_wordcloud.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nWord cloud saved as: {output_path}")
    
    # Get hashtag frequencies
    hashtag_freq = Counter(clean_hashtags).most_common(20)
    print("\nMost common hashtags and their frequencies:")
    for hashtag, freq in hashtag_freq:
        print(f"#{hashtag}: {freq}")

    # Print total number of unique hashtags found
    print(f"\nTotal unique hashtags found: {len(set(clean_hashtags))}")

# Generate the hashtag word cloud
generate_hashtag_wordcloud('data/video_details.csv')
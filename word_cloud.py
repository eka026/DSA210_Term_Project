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

def clean_text(text):
    """Clean and preprocess text data"""
    try:
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Simple word splitting as fallback if NLTK fails
        words = text.split()
        
        try:
            # Try NLTK tokenization
            words = word_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK tokenization failed, using simple splitting. Error: {str(e)}")
        
        # Remove stopwords (in multiple languages since the data appears to have Turkish content)
        try:
            stop_words = set(stopwords.words('english') + stopwords.words('turkish'))
        except Exception:
            # Fallback to a basic set of stopwords if NLTK fails
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            print("Warning: Using basic stopwords list as NLTK stopwords failed to load")
        
        tokens = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text cleaning: {str(e)}")
        return text  # Return original text if cleaning fails

def generate_wordcloud(csv_path, min_word_length=3, max_words=100):
    """Generate and save word cloud from video titles"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {str(e)}")
        return
        
    # Combine all titles and clean the text
    all_titles = ' '.join(df['title'].astype(str))
    clean_titles = clean_text(all_titles)
    
    # Custom color function with colors that work well on white background
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Create a colormap - darker colors for better visibility on white
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
    
    # Create and generate word cloud with simplified parameters
    wordcloud = WordCloud(
        width=1920,           # Standard HD width
        height=1080,          # Standard HD height
        background_color='white',  # White background
        color_func=color_func,
        min_word_length=4,    # Increased minimum word length
        max_words=50,         # Reduced number of words
        collocations=False,   # Disable word pairs for simplicity
        prefer_horizontal=0.8, # More horizontal words
        relative_scaling=0.7,  # Increased size difference between frequencies
        min_font_size=12,     # Slightly larger minimum font size
        max_font_size=220,    # Larger maximum font size
        random_state=42       # For reproducibility
    ).generate(clean_titles)
    
    # Create figure with white background
    plt.style.use('default')  # Use default style instead of dark_background
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the word cloud in the results directory
    output_path = os.path.join('results', 'video_titles_wordcloud.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nWord cloud saved as: {output_path}")
    
    # Get word frequencies for the most common words
    words = clean_titles.split()
    word_freq = Counter(words).most_common(20)
    print("\nMost common words and their frequencies:")
    for word, freq in word_freq:
        print(f"{word}: {freq}")

# Generate the word cloud
generate_wordcloud('data/video_details.csv')
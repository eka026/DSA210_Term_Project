import json
import html
import requests
from urllib.parse import urlparse, parse_qs
import csv
import os


API_KEY = os.getenv("YOUTUBE_API_KEY")

# Function to fetch video category mappings
def fetch_video_categories(api_key):
    url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        "part": "snippet",
        "regionCode": "US",  # Adjust the region code if needed
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return {item["id"]: item["snippet"]["title"] for item in data["items"]}
    else:
        print(f"Error fetching video categories: {response.status_code}")
        return {}

# Load Takeout JSON data
def load_takeout_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        takeout_data = json.load(file)
    return takeout_data

# Function to extract video ID from the titleUrl
def extract_video_id(title_url):
    # Decode the URL
    decoded_url = html.unescape(title_url)
    # Parse the URL and extract the video ID
    parsed_url = urlparse(decoded_url)
    video_id = parse_qs(parsed_url.query).get('v', [None])[0]
    return video_id

# Main function to process videos and save details
def process_videos(takeout_data, api_key):
    # Fetch category mappings
    category_mapping = fetch_video_categories(api_key)

    csv_file = "video_details.csv"
    file_exists = os.path.isfile(csv_file)

    # Load already processed video IDs if the file exists
    processed_video_ids = set()
    if file_exists:
        with open(csv_file, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_video_ids.add(row["video_id"])

    # Open the CSV file in append mode to continue adding data
    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["video_id", "title", "description", "category_id", "category_name",
                      "publish_date", "views", "likes", "dislikes", "channel_title"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file didn't exist before
        if not file_exists:
            writer.writeheader()

        # Iterate over each entry in the Takeout data
        for entry in takeout_data:
            title_url = entry.get("titleUrl")
            if not title_url:
                continue  # Skip if no URL is present

            video_id = extract_video_id(title_url)

            # Skip if we have already processed this video
            if video_id in processed_video_ids:
                print(f"Already processed video ID: {video_id}. Skipping.")
                continue

            if video_id:
                # YouTube Data API endpoint for videos
                url = "https://www.googleapis.com/youtube/v3/videos"
                params = {
                    "part": "snippet,contentDetails,statistics",
                    "id": video_id,
                    "key": api_key
                }
                response = requests.get(url, params=params)

                # Parse the response
                if response.status_code == 200:
                    data = response.json()
                    if "items" in data and len(data["items"]) > 0:
                        video_details = data["items"][0]

                        snippet = video_details.get("snippet", {})
                        statistics = video_details.get("statistics", {})

                        category_id = snippet.get("categoryId", "")
                        category_name = category_mapping.get(category_id, "Unknown Category")

                        # Write the video details to CSV
                        writer.writerow({
                            "video_id": video_id,
                            "title": snippet.get("title", ""),
                            "description": snippet.get("description", ""),
                            "category_id": category_id,
                            "category_name": category_name,
                            "publish_date": snippet.get("publishedAt", ""),
                            "views": statistics.get("viewCount", ""),
                            "likes": statistics.get("likeCount", ""),
                            "dislikes": statistics.get("dislikeCount", ""),  # 'dislikeCount' may not be available
                            "channel_title": snippet.get("channelTitle", "")
                        })
                        processed_video_ids.add(video_id)
                        print(f"Processed video ID: {video_id}")
                    else:
                        print(f"No details found for video ID: {video_id}")
                else:
                    print(f"Error fetching details for video ID {video_id}: {response.status_code}")
            else:
                print("Invalid or missing video ID.")

# Entry point of the script
if __name__ == "__main__":
    # Path to your Takeout JSON file
    takeout_json_file = "watch-history.json"  # Replace with the path to your JSON file

    # Load the Takeout data
    takeout_data = load_takeout_data(takeout_json_file)

    # Process the videos and save details
    process_videos(takeout_data, API_KEY)

    print("Video details have been appended to 'video_details.csv'.")

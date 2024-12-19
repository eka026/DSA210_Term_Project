import json
import html
import requests
import os
import csv
from urllib.parse import urlparse, parse_qs
import re

API_KEY = os.getenv("YOUTUBE_API_KEY")

def iso8601_duration_to_seconds(duration):
    # Parse ISO 8601 duration string (e.g., "PT1H23M45S")
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return hours * 3600 + minutes * 60 + seconds

def fetch_video_categories(api_key):
    url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        "part": "snippet",
        "regionCode": "US",  # Adjust if needed
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return {item["id"]: item["snippet"]["title"] for item in data["items"]}
    else:
        print(f"Error fetching video categories: {response.status_code}")
        return {}

def load_takeout_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        takeout_data = json.load(file)
    return takeout_data

def extract_video_id(title_url):
    decoded_url = html.unescape(title_url)
    parsed_url = urlparse(decoded_url)
    video_id = parse_qs(parsed_url.query).get('v', [None])[0]
    return video_id

def process_videos(takeout_data, api_key):
    category_mapping = fetch_video_categories(api_key)

    csv_file = "video_details.csv"
    file_exists = os.path.isfile(csv_file)

    processed_video_ids = set()
    if file_exists:
        with open(csv_file, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_video_ids.add(row["video_id"])

    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "video_id", 
            "title", 
            "description", 
            "category_id", 
            "category_name",
            "publish_date",
            "channel_title",
            "duration"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for entry in takeout_data:
            title_url = entry.get("titleUrl")
            if not title_url:
                continue

            video_id = extract_video_id(title_url)

            if not video_id:
                print("Invalid or missing video ID.")
                continue

            if video_id in processed_video_ids:
                print(f"Already processed video ID: {video_id}. Skipping.")
                continue

            # Fetch video details using the YouTube Data API
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,contentDetails",
                "id": video_id,
                "key": api_key
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if "items" in data and len(data["items"]) > 0:
                    video_details = data["items"][0]
                    snippet = video_details.get("snippet", {})
                    content_details = video_details.get("contentDetails", {})

                    category_id = snippet.get("categoryId", "")
                    category_name = category_mapping.get(category_id, "Unknown Category")

                    iso_duration = content_details.get("duration", "")
                    duration_seconds = iso8601_duration_to_seconds(iso_duration)
                    duration_hours = duration_seconds / 3600.0

                    writer.writerow({
                        "video_id": video_id,
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "category_id": category_id,
                        "category_name": category_name,
                        "publish_date": snippet.get("publishedAt", ""),
                        "channel_title": snippet.get("channelTitle", ""),
                        "duration": duration_hours
                    })

                    processed_video_ids.add(video_id)
                    print(f"Processed video ID: {video_id}")
                else:
                    print(f"No details found for video ID: {video_id}")
            else:
                print(f"Error fetching details for video ID {video_id}: {response.status_code}")

if __name__ == "__main__":
    takeout_json_file = "watch-history.json"  # Replace with your own file path if needed
    takeout_data = load_takeout_data(takeout_json_file)
    process_videos(takeout_data, API_KEY)
    print("Video details have been appended to 'video_details.csv'.")

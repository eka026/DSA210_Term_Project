# Analyzing My YouTube Usage

For the final report, please click [here](https://sites.google.com/view/termprojectpresentation?usp=sharing).

## Table of Contents
[Introduction and Motivation](#introduction-and-motivation)
[Data Source](#data-source)
[Data Preprocessing](#data-preprocessing)
[Data Visualization](#data-visualization)
[Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
[Machine Learning Experiment](#machine-learning-experiment)
[Findings and Interpretation](#findings-and-interpretation)


## Introduction and Motivation
Social media holds an important place in our lives and is one of the easiest platforms to access for meeting our daily entertainment needs. The same is true for me, as I also spend time on YouTube throughout the day. 

Over the years, the content, categories, and watch durations of the videos I have watched have changed during different periods. Through this project, I analyzed changes in my viewing preferences over time, the factors that influence them, and how my watch durations behaved. This project helped me learn more about myself and provided an opportunity to practice data analysis.

## Tools

## Data Source
The dataset for this project is my personal YouTube watch history, downloaded using **Google Takeout**. 

The dataset includes:
- Video titles
- Channels
- Timestamps
- URLs 

The data is provided in JSON format.

However, this data was not sufficient for me, so I wrote a script called `fetch_data.py` using the YouTube Data API V3. With this code, I fetched all the links and the details of the videos in those links from the `watch-history.json` file (provided by Google Takeout). These details include important information such as the titles of the videos, their durations, channel titles, categories, descriptions, and more. These details were then saved into a `video_details.csv` file for further processing.

## Data Preprocessing
After saving my data to `video_details.csv`, I wrote the `process_data.py` script to work more efficiently with the data. This script takes `watch-history.json` and `video_details.csv` as input and generates the following three files:

- **enhanced_categories.csv**: Contains detailed video categorization, including video ID, category, subcategory, watch time, channel, and duration.
- **temporal_patterns.csv**: Displays viewing patterns over time, breaking down watch duration by month, category, and subcategory.
- **channel_categories.csv**: Provides an analysis of which channels you watch and how much time you spend on each category per channel.

With these three generated files, I reached a level where I could perform data analysis more effectively.

## Data Visualization
In my project, I used data visualization techniques to perform my analyses more effectively. In the `temporal_analysis.py` and `categories_analysis.py` files, I utilized libraries such as Pandas, Matplotlib, and Seaborn to create visualizations like stacked area charts, bar charts, and time series plots. Additionally, in the `word_cloud.py` file, I used the NLTK and WordCloud libraries to generate word cloud visualizations. All of these tools helped me conduct my analyses more effectively.

## Exploratory Data Analysis (EDA)
With the help of data visualization techniques, I conducted a detailed analysis by performing hypothesis testing and statistical analysis on my data. In the `temporal_analysis.py` script, I analyzed my monthly viewing trends, the distribution of my viewing hours across the days of the week, and the time periods during the day when I watched the most. I also compared these findings with each other. Additionally, I analyzed the distribution of my viewing durations across different categories. To achieve this, I utilized tests such as the t-test and Kruskal-Wallis test and saved the relevant results under the `results` and `hypothesis_test_results` directories.

Similarly, in the `categorical_analysis.py` script, I investigated the reasons behind some peaks in my viewing durations and analyzed which categories and channels contributed to those peaks. For this, I used statistical analysis techniques such as the Mann-Whitney U test, two-proportion z-test, Pearson’s Correlation Coefficient, and Gini Coefficient.

## Machine Learning Experiment
In this part of the project, I conducted an experimental machine learning study in the `machine_learning.py` script. This model predicts video categories using video durations and channel names. The script utilizes a Decision Tree Classifier with 5-fold cross-validation to make its predictions. It processes the top 5 most frequent video categories and generates performance metrics, including precision, recall, and F1-scores. The results are visualized through a confusion matrix heatmap, which is stored in the `results` directory.

The model also calculates feature importance to determine whether video duration or channel identity has a greater influence on categorizing videos. This makes the model useful for content categorization and analysis.

## Findings and Interpretation
Analysis of 4.5 years of YouTube viewing data revealed several interesting patterns:

- Watch time varied significantly across months (p < 0.001), contrary to initial assumptions of consistent viewing habits.
- No significant difference was found between weekday and weekend viewing patterns (p = 0.0673), suggesting consistent daily consumption.
- Peak viewing hours occurred between 6-8 PM, with unexpected activity during early morning hours (4-7 AM).

Content preferences showed clear temporal patterns:

- Gaming content dominated during the 2020-2021 pandemic period
- Educational content peaked during university entrance exam preparation periods (2021-2022)
- Financial content viewing strongly correlated with market performance (correlation coefficient: 0.8069)
- Recent shift towards programming education content (2023-2024)

- Category distribution was highly skewed (Gini coefficient: 0.749), with Education and Gaming categories accounting for the majority of watch time.

## Limitations and Future Work

### Limitations
- YouTube Data API V3 Limitations:
 - No access to partial watch duration data - videos are recorded as watched even if only partially viewed.
 - Historical data limited to 4.5 years, preventing longer-term analysis.

- Category classifications are broad and may not capture nuanced content differences.

### Future Work
- Explore correlations with other life events and external factors.
- Implement watch percentage tracking to better understand content engagement.
- Include analysis of interaction patterns (likes, comments).
- Integrate cross-platform analysis to understand overall digital content consumption.


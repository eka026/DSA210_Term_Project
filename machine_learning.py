import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Read the data
df = pd.read_csv('data/enhanced_categories.csv')

# Select top 5 categories by number of videos
top_categories = df['category'].value_counts().nlargest(5).index
df_filtered = df[df['category'].isin(top_categories)].copy()

# Encode channel names
label_encoder = LabelEncoder()
df_filtered['channel_encoded'] = label_encoder.fit_transform(df_filtered['channel'])

# Prepare features
X = df_filtered[['duration', 'channel_encoded']]
y = df_filtered['category']

# Initialize model
model = DecisionTreeClassifier(max_depth=6, random_state=42)

# Perform k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Get cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=kf)

# Get predictions for confusion matrix
y_pred = cross_val_predict(model, X, y, cv=kf)

# Print cross-validation results
print("\nCross-validation scores:", cv_scores)
print(f"Average CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Create classification report as DataFrame
report = classification_report(y, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).round(3)

# Reorder and rename columns
report_df = report_df.T.reset_index()
report_df.columns = ['Category', 'Precision', 'Recall', 'F1-score', 'Support']
report_df = report_df[~report_df['Category'].isin(['accuracy', 'macro avg', 'weighted avg'])]

# Print formatted table
print("\nClassification Report:")
print("\n" + report_df.to_string(index=False))

# Also print the overall metrics
print("\nOverall Metrics:")
overall_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro Avg', 'Weighted Avg'],
    'Value': [
        report['accuracy'],
        np.mean([report[cat]['f1-score'] for cat in top_categories]),
        report['weighted avg']['f1-score']
    ]
}).round(3)
print("\n" + overall_metrics.to_string(index=False))

# Create and save confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_categories,
            yticklabels=top_categories)
plt.title('Confusion Matrix (Cross-validated)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/confusion_matrix_cv.png')
plt.close()

# Train final model on full dataset
model.fit(X, y)

# Print feature importance
feature_names = ['duration', 'channel']
print("\nFeature Importances:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Print some channel information
print("\nSample of Channel Encoding:")
sample_channels = pd.DataFrame({
    'channel': label_encoder.classes_[:5],  # Show first 5 channels
    'encoded_value': range(5)
})
print(sample_channels)
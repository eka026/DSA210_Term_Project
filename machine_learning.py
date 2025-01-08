import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create results directory if it doesn't exist
Path("results").mkdir(parents=True, exist_ok=True)

def prepare_features():
    """Prepare features for classification without using subcategory."""
    df = pd.read_csv('data/enhanced_categories.csv')
    
    features = pd.DataFrame()
    
    # Time-based features
    df['watched_on'] = pd.to_datetime(df['watched_on'], format='ISO8601')
    features['hour'] = df['watched_on'].dt.hour
    features['day_of_week'] = df['watched_on'].dt.dayofweek
    features['month'] = df['watched_on'].dt.month
    
    # Video features
    features['duration'] = df['duration']
    
    # Channel encoding
    le_channel = LabelEncoder()
    features['channel_encoded'] = le_channel.fit_transform(df['channel'])
    
    # Target variable
    le_category = LabelEncoder()
    target = le_category.fit_transform(df['category'])
    
    return features, target, le_category.classes_

def plot_confusion_matrix(y_test, y_pred, class_names):
    """Create and save a detailed confusion matrix visualization."""
    plt.figure(figsize=(20, 15))
    
    # Create confusion matrix and normalize it
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap with improved readability
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',  # Changed colormap for better visibility
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,  # Make cells square
        cbar_kws={'label': 'Proportion of Predictions'}
    )
    
    # Customize the plot
    plt.title('Content Category Prediction Confusion Matrix\n(Normalized by True Category)', 
              pad=20, size=16)
    plt.xlabel('Predicted Category', size=12, labelpad=10)
    plt.ylabel('True Category', size=12, labelpad=10)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high DPI for clarity
    plt.savefig('results/confusion_matrix_revised.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(clf, feature_names):
    """Plot feature importance without subcategory."""
    plt.figure(figsize=(10, 6))
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title('Feature Importance in Category Prediction', pad=20)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('results/feature_importance_revised.png', dpi=300)
    plt.close()

def main():
    print("Preparing features (without subcategory)...")
    features, target, class_names = prepare_features()
    
    print("Training and evaluating model...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    y_pred = clf.predict(X_test_scaled)
    
    print("\nModel Performance:")
    print(f"Training Accuracy: {train_score:.3f}")
    print(f"Testing Accuracy: {test_score:.3f}")
    print(f"Difference: {abs(train_score - test_score):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nCreating visualizations...")
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_feature_importance(clf, features.columns)
    
    print("\nMost Important Features:")
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    print("\nVisualizations have been saved to the 'results' folder:")
    print("- confusion_matrix_revised.png")
    print("- feature_importance_revised.png")

if __name__ == "__main__":
    main()
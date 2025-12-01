# ============================================================================
# FILE: src/train_model.py
# ============================================================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# Get project root and models directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load Pima Indians Diabetes dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    df = pd.read_csv(url, names=column_names)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"\nClass distribution:")
    print(f"  Non-Diabetic (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.2f}%)")
    print(f"  Diabetic (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.2f}%)")
    
    return X, y, column_names[:-1]

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier with optimized hyperparameters"""
    print("\n🌲 Building Random Forest Classifier...")
    
    # Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,        # Number of trees
        max_depth=10,            # Maximum depth of trees
        min_samples_split=10,    # Minimum samples to split a node
        min_samples_leaf=4,      # Minimum samples in leaf node
        max_features='sqrt',     # Number of features to consider for best split
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("\nModel Configuration:")
    print(f"  • Number of trees: {rf_model.n_estimators}")
    print(f"  • Max depth: {rf_model.max_depth}")
    print(f"  • Min samples split: {rf_model.min_samples_split}")
    print(f"  • Class weight: {rf_model.class_weight}")
    
    return rf_model

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 DIABETES PREDICTION - RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    
    print("\n[1/6] Loading and preparing data...")
    X, y, feature_names = load_and_prepare_data()
    
    print("\n[2/6] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n[3/6] Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    
    # Train the model
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    print("\n[4/6] Performing Cross-Validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n[5/6] Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*70}")
    print("📊 MODEL PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    
    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Non-D  Diabetic")
    print(f"Actual Non-D     {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       Diabetic  {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    print(f"\n[6/6] Feature Importance Analysis...")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🎯 FEATURE IMPORTANCE (Most to Least Important):")
    print(f"{'='*70}")
    for idx, row in feature_importance.iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"{row['Feature']:25s} | {bar} {row['Importance']:.4f}")
    
    print(f"\n{'='*70}")
    print("💾 Saving model to models/ directory...")
    
    # Save to models directory
    model_path = os.path.join(MODEL_DIR, 'diabetes_rf_model.pkl')
    feature_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(feature_names, feature_path)
    
    print(f"{'='*70}")
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Feature names saved to: {feature_path}")
    print(f"{'='*70}")
    
    # Make sample predictions
    print("\n🧪 Testing sample predictions...")
    sample_indices = [0, 10, 20]
    for idx in sample_indices:
        sample = X_test.iloc[idx:idx+1]
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]
        actual = y_test.iloc[idx]
        
        print(f"\nSample {idx}:")
        print(f"  Actual: {'Diabetic' if actual == 1 else 'Non-Diabetic'}")
        print(f"  Predicted: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"  Probability: Non-Diabetic={probability[0]:.2%}, Diabetic={probability[1]:.2%}")
        print(f"  Result: {'✅ Correct' if prediction == actual else '❌ Incorrect'}")
    
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
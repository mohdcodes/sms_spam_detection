# scripts/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='latin-1')

# Preprocess data
def preprocess_data(data):
    data = data.rename(columns={"v1": "label", "v2": "text"})
    data = data[['label', 'text']]
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    X = data['text']
    y = data['label']
    return X, y

# Evaluate model
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    return accuracy, report

if __name__ == "__main__":
    data = load_data('../data/spam.csv')
    X, y = preprocess_data(data)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load('../models/spam_detector.pkl')
    accuracy, report = evaluate_model(model, X_val, y_val)
    
    print(f'Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')

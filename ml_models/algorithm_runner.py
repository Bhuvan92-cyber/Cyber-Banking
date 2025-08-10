import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def encode_and_split(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X, y = SMOTE().fit_resample(X, y)
    return train_test_split(X, y, test_size=0.3, random_state=42)

def run_all_algorithms(df):
    X_train, X_test, y_train, y_test = encode_and_split(df)

    models = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'report': report,
            'matrix': matrix
        }

    return results
 

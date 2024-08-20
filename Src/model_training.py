import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def train_and_evaluate(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = [
        MultinomialNB(),
        LogisticRegression(max_iter=1000),
        LinearSVC(),
        RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200, random_state=42),
        GradientBoostingClassifier(),
        SGDClassifier(max_iter=1000)
    ]
    
    accuracy_scores = []
    
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_scores.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    model_names = [type(model).__name__ for model in models]
    plt.bar(model_names, accuracy_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.title('Model Accuracies')
    plt.show()
    
    return accuracy_scores

if __name__ == "__main__":
    X = pd.read_csv('/content/engineered_train_data.csv').values
    y = pd.read_csv('/content/engineered_train_labels.csv').values.ravel()
    accuracy_scores = train_and_evaluate(X, y)
    print(accuracy_scores)

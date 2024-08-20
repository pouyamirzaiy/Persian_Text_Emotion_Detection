import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def perform_grid_search(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    return best_model

if __name__ == "__main__":
    X = pd.read_csv('/content/engineered_train_data.csv').values
    y = pd.read_csv('/content/engineered_train_labels.csv').values.ravel()
    best_model = perform_grid_search(X, y)
    print(best_model)

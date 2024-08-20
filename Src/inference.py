import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def load_model_and_vectorizer():
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    model = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200, random_state=42)
    return model, vectorizer

def make_predictions(df_test, model, vectorizer):
    X_test = vectorizer.transform(df_test['Text'])
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    df_test = pd.read_csv('/content/3rdHW_test.csv', header=None, names=["Text", "Emotion"])
    model, vectorizer = load_model_and_vectorizer()
    predictions = make_predictions(df_test, model, vectorizer)
    df_test['Predicted_Emotion'] = predictions
    df_test.to_csv('/content/test_predictions.csv', index=False)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def feature_engineering(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['Text'])
    y = df['Emotion']
    
    return X, y, vectorizer

if __name__ == "__main__":
    df = pd.read_csv('/content/cleaned_train_data.csv')
    X, y, vectorizer = feature_engineering(df)
    pd.DataFrame(X.toarray()).to_csv('/content/engineered_train_data.csv', index=False)
    pd.Series(y).to_csv('/content/engineered_train_labels.csv', index=False)

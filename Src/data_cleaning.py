import pandas as pd
from hazm import Normalizer, Stemmer, word_tokenize, stopwords_list, POSTagger

def clean_data(df):
    # Initialize the normalizer, stemmer, and POS tagger
    normalizer = Normalizer()
    stemmer = Stemmer()
    tagger = POSTagger(model='resources/postagger.model')
    
    # Get the list of Persian stop words
    stopwords = stopwords_list()
    
    # Function to clean, stem, and add POS tags to the text
    def clean_text(text):
        # Normalize the text
        text = normalizer.normalize(text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words and stem the tokens
        tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
        
        # Add POS tags
        pos_tags = tagger.tag(tokens)
        tagged_tokens = [f'{word}_{tag}' for word, tag in pos_tags]
        
        return ' '.join(tagged_tokens)
    
    # Apply the function to the 'Text' column
    df['Text'] = df['Text'].apply(clean_text)
    
    return df

if __name__ == "__main__":
    df = pd.read_excel('/content/train_data.xlsx', header=None, names=["Text", "Emotion"])
    df = clean_data(df)
    df.to_csv('/content/cleaned_train_data.csv', index=False)

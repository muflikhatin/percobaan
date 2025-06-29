import nltk
import pandas as pd
import streamlit as st
import re, string
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from itertools import chain
from indoNLP.preprocessing import pipeline, replace_word_elongation, replace_slang, emoji_to_words, remove_html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
def main():

    # Download resources
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Load dataset
    df_teroris = pd.read_csv('indonesian_tweet_about_teroris_(200001-210251).csv')
    df_teroris = df_teroris.head(500)
    df_teroris.drop_duplicates(inplace=True)
    df_teroris = df_teroris.drop(['tweet id','username', 'reference type', 'reference id', 'created at', 'like', 'quote', 'reply', 'retweet', 'tweet url', 'mentions', 'hashtags'], axis=1)

    # Text Cleaning
    def cleaning(text):
        text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('nan', '', text)
        return text

    def preprocess_data(df):
        # Preprocess text
        df['tweet text (clean)'] = df['tweet text'].apply(lambda x: pipe(x))
        df['tweet text (clean)'] = df['tweet text (clean)'].apply(lambda x: cleaning(x))
        df['tweet text (clean)'] = df['tweet text (clean)'].replace('', np.nan)
        df.dropna(inplace=True)
        
        # Labeling tweets
        df['label'] = df['tweet text'].apply(lambda x: label_tweet(x))
        
        # Tokenizing tweets
        df['token'] = df['tweet text (clean)'].apply(lambda x: tknzr.tokenize(x))
        
        # Removing stopwords
        stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
        df['token'] = df['token'].apply(lambda x: [w for w in x if not w in stop_words])
        
        return df

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()

    def label_tweet(tweet):
        score = sia.polarity_scores(tweet)
        if score['compound'] >= 0.05:
            return 1  # tweet bernilai positif
        elif score['compound'] <= -0.05:
            return 2  # tweet bernilai negatif
        else:
            return 0  # tweet bernilai netral

    # Initialize tokenizer
    tknzr = TweetTokenizer()

    # Initialize preprocessing pipeline
    pipe = pipeline([replace_word_elongation, replace_slang, emoji_to_words, remove_html])

    # Preprocess data
    @st.cache()
    def preprocess_data_cached(df):
        return preprocess_data(df)
        # return df['cleaned_tweet']
    def select_features(data, labels, num_features=1000):
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(data)
        
        # Calculate information gain scores
        info_gain = mutual_info_classif(features, labels)
        
        # Select top k features with highest scores
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selected_features = selector.fit_transform(features, labels)
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        
        # Get feature names
        feature_names = np.array(vectorizer.get_feature_names_out())[selected_indices]
        # feature_names = np.array(vectorizer.get_feature_names())[selected_indices]

        
        return selected_features, feature_names
    # Streamlit App
    st.title("Analisis Sentimen dan Pemodelan")
    st.write(" ")
    st.header("Dataset Sebelum Dilakukan Preprocessing")
    st.dataframe(df_teroris)

    st.header("Data Hasil Preprocessing")
    if st.button("Preprocessing Data"):
        processed_data = preprocess_data_cached(df_teroris)
        st.success("Preprocessing data selesai.")
        st.dataframe(processed_data)

    st.header("Analisis Sentimen")
    sentiment_text = st.text_input("Masukkan teks untuk analisis sentimen:")
    
    if sentiment_text:
        sentiment_prediction = label_tweet(sentiment_text)
        sentiment_label = "Positif" if sentiment_prediction == 1 else "Negatif" if sentiment_prediction == 2 else "Netral"
        st.success(f"Sentimen: {sentiment_label}")

    st.header("Performa Model")
    accuracy = 0.0
    precision = 0.0
    recall = 0.0

    if st.button("Evaluasi Model"):
        # Splitting the data
        processed_data = preprocess_data_cached(df_teroris)
       
        # memisahkan text dan label
        texts = processed_data['tweet text (clean)'].values
        labels = processed_data['label'].values
        
        # Perform Feature Selection using Information Gain
        selected_features, _ = select_features(texts, labels)
        
        # Pembagian dataset menjadi data latih dan data uji
        texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        features_train = vectorizer.fit_transform(texts_train)
        features_test = vectorizer.transform(texts_test)
        features_train = features_train.toarray()
        features_test = features_test.toarray()
        features_train = features_train.reshape(features_train.shape[0], features_train.shape[1], 1)
        features_test = features_test.reshape(features_test.shape[0], features_test.shape[1], 1)

        # Model Training
        model = Sequential()
        model.add(LSTM(128, input_shape=(features_train.shape[1], features_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(features_train, labels_train, epochs=5, batch_size=32)

        # Model Evaluation
        predictions = model.predict(features_test)
        predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(labels_test, predictions)
        precision = precision_score(labels_test, predictions, average='weighted')
        recall = recall_score(labels_test, predictions, average='weighted')

    st.write("Akurasi:", accuracy)
    st.write("Presisi:", precision)
    st.write("Recall:", recall)
    
    st.header("Tabel Uji Coba")
    data = {
        'Method': [
            "Tanpa Oversampling",
            "Tanpa Oversampling",
            "Tanpa Oversampling",
            "Meyamakan antara kelas minoritas dengan kelas mayoritas",
            "Meyamakan antara kelas minoritas dengan kelas mayoritas",
            "Meyamakan antara kelas minoritas dengan kelas mayoritas dan menggunakan K-fold",
            "Dengan Oversampling Smote"
        ],
        'Accuracy': [
            "92.01%",
            "91.27%",
            "92.50%",
            "86.00%",
            "0.92%",
            "92.62%",
            "0.86%"
        ],
        'Epoch': [20, 25, 3, 3, 50, 3, 20],
        'Test Size': [0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2],
        'Random State': [42, 42, 60, 60, 42, 60, 42]
    }

    # Buat DataFrame
    df = pd.DataFrame(data)

    # Tampilkan tabel di Streamlit
    st.table(df)
if __name__ == "__main__":
    main()

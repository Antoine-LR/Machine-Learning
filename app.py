import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback


app = Flask(__name__)

path = "C:/Users/perso/P5_ML_PROJECT/"
pipeline = joblib.load(path + "pipeline.pkl")

with open('words_importante.pkl', 'rb') as f:
    words_importante = pickle.load(f)

with open('tags_importante.pkl', 'rb') as f:
    tags_importante = pickle.load(f)
    
with open('multilabel_binarizer.pkl', 'rb') as f:
    multilabel_binarizer = pickle.load(f)    
    
def clean_text(text):
    text = re.sub(r"\'", " ", text) 
    text = re.sub(r"\n", " ", text) 
    text = re.sub(r"\xa0", " ", text) 
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text
#__________________

charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'

def remove_non_alphabetical_character(text):
    """remove all non-alphabetical character"""
    text = re.sub("[^a-z]+", " ", text) # remove all non-alphabetical character
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    return text
#__________________

token = ToktokTokenizer()
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    """remove common words in english by using nltk.corpus's list"""
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered)) # Return the text untokenize

#__________________

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    """Lemmatize the text by using tag """
    
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))  # tokenize the text then return a list of tuple (token, nltk_tag)
    lemmatized_text = []
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'a')) # Lemmatisze adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'v')) # Lemmatisze verbs
        elif tag.startswith('N'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'n')) # Lemmatisze nouns
        elif tag.startswith('R'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'r')) # Lemmatisze adverbs
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatization
    return " ".join(lemmatized_text) # Return the text untokenize
#__________________

def filter_words(words):
    filtered_words = list(filter(lambda w: w in words_importante, words))
    if len(filtered_words) == 0:
        return np.nan
    else:
        # retourne les words filtrés
        return ' '.join(filtered_words)
#__________________

def filter_tags(tags):
    # filtre les tags en gardant uniquement ceux qui sont importants
    filtered_tags = [tag for tag in tags if tag in tags_importante]
    if len(filtered_tags) == 0:
        return np.nan
    else:
        # retourne une tuple contenant les tags filtrés et leur longueur
        return ' '.join(filtered_tags)
#__________________
def add_brackets(s):
    if isinstance(s, float) and np.isnan(s):
        return "[]"
    else:
        return '[' + str(s) + ']'

#__________________
@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    f=request.files['upload_file']
    filename = f.filename
    f.save(filename)
    if pipeline: #si le model existe
        try:
            data=pd.read_csv(filename) #lis le fichier 
            # Cleaning data
            data['Body'] = data['Body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
            data['Title'] = data['Title'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
            data['Title'] = data['Title'].apply(lambda x: clean_text(x)) 
            data['Body'] = data['Body'].apply(lambda x: clean_text(x)) 
            data['Title'] = data['Title'].str.lower()
            data['Body'] = data['Body'].str.lower()
            data['Title'] = data['Title'].apply(lambda x: remove_non_alphabetical_character(x))
            data['Body'] = data['Body'].apply(lambda x: remove_non_alphabetical_character(x)) 
            data['Title'] = data['Title'].apply(lambda x: remove_stopwords(x))
            data['Body'] = data['Body'].apply(lambda x: remove_stopwords(x))
            data['Title'] = data['Title'].apply(lambda x: lemmatize_text(x))
            data['Body'] = data['Body'].apply(lambda x: lemmatize_text(x))
            # tags
            tokenizer = nltk.RegexpTokenizer(r'[<>]', gaps=True)
            data['Tags_clean'] = data['Tags'].map(tokenizer.tokenize)
            data['Tags_filtered'] = data['Tags_clean'].apply(lambda x: filter_tags(x))
            data = data.dropna(subset=['Tags_filtered'])
            data['Tags'] = data['Tags_filtered'].str.replace(' ', ', ')
            data['Tags'] = data['Tags'].apply(lambda x: x.split(",")).tolist()
            # Fusion
            data['Texte_clean'] = data['Title'] + ' ' + data['Body'] 
            # dictionnaire de mots
            data['Text_split'] = data.apply(lambda r :r['Texte_clean'].split(), axis=1)
            # Application de la fonction
            data['Words_filtered'] = data['Text_split'].apply(lambda x: filter_words(x))
            data = data.dropna(subset=['Words_filtered'])
            data['Text'] = data['Words_filtered'].str.replace(' ', ', ')
            data['Text'] = data['Text'].apply(add_brackets)
            # model prediction
            x = data['Text']
            y = data['Tags']
            y_binarized = multilabel_binarizer.transform(y)
            
            # Entraîner le pipeline
            threshold = 0.13
            y_pred = pipeline.predict_proba(x)
            y_pred_inversed=multilabel_binarizer.inverse_transform((y_pred>threshold)*1)
            y_inversed = multilabel_binarizer.inverse_transform(y_binarized)
            
            # Créer un dataframe pour stocker les résultats
            tags = list(map(str, y_inversed))
            pred = list(map(str, y_pred_inversed))

            # Créer un dictionnaire en zippant les deux listes
            df = pd.DataFrame(list(zip(tags, pred)), columns=['Tags', 'Predictions'])
            dictionary = df.to_dict(orient='records')

            # Retourner le dictionnaire sous forme de JSON
            return jsonify(dictionary)

        except:
            return jsonify({'trace':traceback.format_exc()})
        else:
            print('Train the model first')
            return('No model here to use')
                

if __name__ == '__main__':
    
    print('Model loaded')
    app.run(host='localhost', port=5000, debug=True) #localhost
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# removing url
def remove_url(text):
    text = re.sub(r"https?://\S+","",text)
    return text

# removing punctuations
def remove_punc(text):
    text=re.sub(r"[^A-Za-z0-9\s]","",text)
    return text

# removing html tags
def remove_html(text):
    text=re.sub(r"<.*?>","",text)
    return text

# removing stop words
def remove_stopwords(text):
    tokens=word_tokenize(text)
    stop_words = stopwords.words("english")
    for i in tokens:
        if i in stop_words:
            text = text.replace(i,"")

    return text    


# stemming
from nltk.stem import PorterStemmer

def stemming(text):
    ps=PorterStemmer()
    stem_words=[]
    tokens = word_tokenize(text)
    for words in tokens:
        stemmed_tokens = ps.stem(words)
        stem_words.append(stemmed_tokens)

    return " ".join(stem_words)    

def preprocess(text):
    text = text.lower()
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punc(text)
    text = remove_stopwords(text)
    text = stemming(text)

    return text

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

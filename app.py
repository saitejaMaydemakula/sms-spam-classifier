import streamlit as st
import pickle
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('punkt_tab')
nltk.download('stopwords')
def lower_case(sentence):
    sent=sentence.lower()
    sent=nltk.word_tokenize(sent)
    rem=[]
    for i in sent:
        if i.isalnum() and i not in stopwords.words('english'):
            var=ps.stem(i)
            rem.append(var)
    return " ".join(rem)
st.title('SMS Spam Classifier')
input=st.text_area('Enter your message')
if st.button('Classify'):
    transform = lower_case(input)
    vector=tfidf.transform([transform])
    result=model.predict(vector)[0]
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
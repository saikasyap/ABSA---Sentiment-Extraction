
# # Libraries 
import os
from tensorflow.keras.models import Model
from pyabsa import ATEPCCheckpointManager, available_checkpoints
os.environ['PYTHONIOENCODING'] = 'UTF8'
import pandas as pd
import sys
import nltk
import ast
import re
import itertools
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from autocorrect import Speller
import  emoji
from textblob import TextBlob
import spacy
from ast import literal_eval
from numpy.core.defchararray import add
import findfile
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import streamlit as st
import stanza
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
stop_words = set("english")
nlp = stanza.Pipeline('en')
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pd.get_option("display.max_columns")

import classy_classification


#VADER Sentiment Analysis

def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    # print("Overall sentiment dictionary is : ", sentiment_dict)
    # print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    # print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    # print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
    #
    # print("Sentence Overall Rated As", end = " ")
    #
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"
        # print("Positive")

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"
        # print("Negative")

    else :
        overall_sentiment = "Neutral"
        # print("Neutral")
    #
    return overall_sentiment




# Review Summarization 
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
def summarizer(Sentence):
    tokens = tokenizer(Sentence, truncation=True, padding="longest", return_tensors="pt")
    summary = model.generate(**tokens)
    return tokenizer.decode(summary[0])



# Aspect Based Sentiment Analysis Quadraplet Extraction .................
# Preprocessing -----------------------------------------------------------------------


#Spell Check coorect word 
def correct_word (word):
    spell = Speller(lang='en')
    return spell(word)
def undo_contractions(phrase):
    # specific
    phrase = re.sub(r"won[\'’]t", "will not", phrase)
    phrase = re.sub(r"can[\'’]t", "can not", phrase)
    # general
    phrase = re.sub(r"n[\'’]t", " not", phrase)
    phrase = re.sub(r"[\'’]re", " are", phrase)
    phrase = re.sub(r"[\'’]s", " is", phrase)
    phrase = re.sub(r"[\'’]d", " would", phrase)
    phrase = re.sub(r"[\'’]ll", " will", phrase)
    phrase = re.sub(r"[\'’]t", " not", phrase)
    phrase = re.sub(r"[\'’]ve", " have", phrase)
    phrase = re.sub(r"[\'’]m", " am", phrase)
    return phrase



def aspect_sentiment_analysis(txt, stop_words, nlp):
    xt = txt.lower() # LowerCasing the given Text
    sentList = nltk.sent_tokenize(txt) # Splitting the text into sentences

    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic = {}

    for line in sentList:
        newtaggedList = []
        txt_list = nltk.word_tokenize(line) # Splitting up into words
        taggedList = nltk.pos_tag(txt_list) # Doing Part-of-Speech Tagging to each word

        newwordList = []
        flag = 0
        for i in range(0,len(taggedList)-1):
            if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"): # If two consecutive words are Nouns then they are joined together
                newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                flag=1
            else:
                if(flag==1):
                    flag=0
                    continue
                newwordList.append(taggedList[i][0])
                if(i==len(taggedList)-2):
                    newwordList.append(taggedList[i+1][0])

        finaltxt = ' '.join(word for word in newwordList) 
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stop_words]
        taggedList = nltk.pos_tag(wordsList)

        doc = nlp(finaltxt) # Object of Stanford NLP Pipeleine
        
        # Getting the dependency relations betwwen the words
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

        # Coverting it into appropriate format
        for i in range(0, len(dep_node)):
            if (int(dep_node[i][1]) != 0):
                dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

        featureList = []
        categories = []
        for i in taggedList:
            if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
                featureList.append(list(i)) # For features for each sentence
                totalfeatureList.append(list(i)) # Stores the features of all the sentences in the text
                categories.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                    if(j[0]==i[0]):
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i[0], filist])
            
    for i in totalfeatureList:
        dic[i[0]] = i[1]
    
    for i in fcluster:
        if(dic[i[0]]=="NN"):
            finalcluster.append(i)
        
    return(finalcluster)

def absa_triple(Sentence, data):
    sent = Sentence.replace(r'([a-z]+)([A-Z])', r'\1 \2').lower().replace('\n', '.').replace(r'\s*\.+\s*', '. ') .replace(r'([\{\(\[\}\)\]])', r' \1 ') .replace(r'([:])', r' \1 ').replace(r'(\d+\.?\d*)', r' \1 ')
    sent = undo_contractions(sent)
    clean_text = correct_word(sent)
    nlp = spacy.load("en_core_web_sm")
    #Convert in to list of sentences....................................
    sentences = [i.text.strip() for i in nlp(clean_text).sents]


    # Applying PyABSA Model

    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual',
                                                               auto_device=True  # False means load model on CPU
                                                               )


    def inference(text):
        result = aspect_extractor.extract_aspect(inference_source=[text],
                                             pred_sentiment=True)

        result = pd.DataFrame({
        'aspect': result[0]['aspect'],
        'sentiment': result[0]['sentiment']
        })

        return result


    result1 = aspect_extractor.extract_aspect(inference_source=sentences,
                                          save_result=False,
       
                                          print_result=True,
                                          pred_sentiment=True)



    result = pd.DataFrame(result1)
    result = result[['sentence', 'aspect', 'sentiment']]

    # Converting the resultant Dataframe in to a certain format.. 


    def convert(df):
        df['sentence'] = df['sentence'].astype('str') 
        df['aspect'] = df['aspect'].apply(lambda x: np.array(x))
        df['sentiment'] = df['sentiment'].apply(lambda x: np.array(x))
        df =df.dropna()
 
        df['aspect'] = df['aspect'].values.tolist()
        df['sentiment'] = df['sentiment'].values.tolist()

        df['tup'] = df.apply(lambda x: list(zip(x.aspect,x.sentiment)), axis=1)
        df['tup'] = df['tup'].astype('str').apply(literal_eval)
        df2 =df.explode('tup')
        df2 = df2.drop(['aspect', 'sentiment'], 1)
        df2['aspect'], df2['aspect_sentiment'] = df2.tup.str
        df2 = df2.drop_duplicates()
        df2 =df2.drop('tup', axis=1)

        return df2


    #Exploding Aspects
    df2 = convert(result)
    filtered_df = df2[df2['aspect'].notnull()].reset_index(drop = True)
    filtered_df['sent_index'] = filtered_df.groupby('sentence', sort=False).ngroup() + 1

    # List of sentences and aspects
    n_asp = filtered_df['sentence'].tolist()
    asp = filtered_df['aspect'].tolist()

    n_sp = []
    for query in n_asp:
        query1 = re.sub(r'[^ \w+]', '', query).replace("_", " ").replace("+"," ").strip()
        n_sp.append(query1)



    # Opinion Term Extraction ..........
    l_sup = []
    for i in n_sp:
        try:
            l_sup.append(aspect_sentiment_analysis(i, stop_words, nlp))
            print(aspect_sentiment_analysis(i, stop_words, nlp))
        except:
            l_sup.append([])



    dct ={}
    for i in range(0,len(l_sup)):
        lst1 = l_sup[i]
        for j in range(0,len(lst1)):
            if asp[i].replace(" ", "") == lst1[j][0] or asp[i] in lst1[j][0]:
                dct[i] = lst1[j][1]


    for i in range(len(l_sup)):
        if i not in dct:
            dct[i] = []

    sd = {k : dct[k] for k in sorted(dct)}
    opin = list(sd.values())

    filtered_df['opinions'] =  opin

    ###Aspect Category Detection ..................................
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data": data,
            "model": "typeform/distilbert-base-uncased-mnli",
            "cat_type": "zero",
        }
    )


    lst1 = []
    for i in n_asp:
        dct = nlp(i)._.cats
        keyMax =max(zip(dct.values(), dct.keys()))[1]

        lst1.append(keyMax)


    filtered_df['aspect_category'] = lst1

    return(filtered_df)




STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """
# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
def main():

    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Sentiment Engine</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    # Collect Input from user :

    if Review_type == "Game":
        data = ["Gameplay", "Performance Issues", "Other", "Narrative", "Graphics", "Level design", "Multiplayer", "Value for Money", "Replay Value"]
    else:
        data = ["Food","Service", "Ambiance", "Other"] 

    query = str()
    query = str(st.text_input("Enter the review you want to analyse(Press Enter once done)"))
    if len(query) > 0:
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Search Results are being extracted"):
            df = absa_triple(query,data)
        #print(lst['Results'])
        st.dataframe(df)
        st.write("Overall Sentiment of the query : {} ".format(sentiment_vader(query)))
        st.write("Summarizing the query : {} ".format(summarizer(query)))

        st.write("Length of the query '{}' are : {} ".format(query,len(query)))
        st.write("Total Positive aspects are : {}".format(len(df[df["aspect_sentiment"]=="positive"])))
        st.write("Total Negative aspects are : {}".format(len(df[df["aspect_sentiment"]=="negative"])))
        st.write("Total Neutral aspects are : {}".format(len(df[df["aspect_sentiment"]=="neutral"])))
       
        st.success('Search results have been extracted !!!!')

    if st.button("Exit"):
        st.balloons()


if __name__ == '__main__':
    main()
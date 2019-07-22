# Modules
import pickle
import praw
import requests 
import numpy as np
import pandas as pd
import json
import time
import string
import os

# NLP PreProcessors
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Config env
clientID = os.environ.get('CLIENT_ID', 'None')
clientSECRET = os.environ.get('CLIENT_SECRET', 'None')

# Praw Session Initialized
reddit = praw.Reddit(client_id=clientID,
                     client_secret=clientSECRET,
                     user_agent='default')

# Trained Models
pickle_in_Model = open("./trainedData/trainedModel.pickle","rb")
pickle_in_Vectorizer = open("./trainedData/vectorizer.pickle","rb")
pickle_in_Labels = open("./trainedData/labels.pickle","rb")

# Loaded
model = pickle.load(pickle_in_Model)
tfidVectorizer = pickle.load(pickle_in_Vectorizer)
labels = pickle.load(pickle_in_Labels)

# Labels to CSS Class
labelToClass = {
    'AskIndia': 'ask',
    'Non-Political': 'non_P',
    '[R]eddiquette': 'reddiquette',
    'Schduled': 'scheduled',
    'Photography': 'photo',
    'Science/Technology': 'science',
    'Politics': 'politics',
    'Buisiness/Finance': 'buisness',
    'Policy/Economy': 'buisness',
    'Sports': 'sports',
    'Food': 'food'
}

# Text Pre-Processing
def cleanText(inputText):
    if(type(inputText)==float):
        inputText = ''
    inputText = str((inputText.encode('ascii', 'ignore')).decode('utf-8')).lower().split()
    specialChars = string.punctuation.replace('#','').replace('+','').replace('_','')
    table = str.maketrans('', '', specialChars)
    words = [w.translate(table) for w in inputText]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    words = ' '.join(stemmed)
    return words

def splitUrl(inputText):
    inputText = inputText.lower().split('/')
    inputText = filter(None, inputText)
    inputText = [x for x in inputText if ((x != 'https:') and (x != 'http:'))]
    inputText[0] = inputText[0].split('.')
    inputText[0] = [x for x in inputText[0] if ((x != 'com') and (x != 'www'))]
    inputText[0] = ' '.join(inputText[0])
    words = ' '.join(inputText)
    return words

def classifyTime(inputText):
    hours = time.localtime(int(inputText)).tm_hour
    if(hours in range(6,12)):
        return 'Morning'
    elif(hours in range(12, 17)):
        return 'Noon'
    elif(hours in range(17, 21)):
        return 'Evening'
    else:
        return 'Night'

# Get extract inputs from link
def getPostDetails(redditUrl, dataDictionary):
    try:
        post = reddit.submission(url=redditUrl)
    except:
        return False
    
    url = 'https://api.pushshift.io/reddit/submission/search/?subreddit=india&ids=' + str(post.id)
    req = json.loads(requests.get(url).text)
    
    if(len(req['data']) != 1):
        return False
    
    commentText = ''
    post.comments.replace_more(limit=0)
    comments = post.comments.list()
    for comment in comments:
        if(comment.is_root):
            commentText += str(comment.body)+' '
    
    submission = req['data'][0]
    dataDictionary['author_fullname'].append(str(submission.setdefault('author_fullname', 'null')))
    dataDictionary['created_utc'].append(submission.setdefault('created_utc', 0))
    dataDictionary['domain'].append(str(submission.setdefault('domain', 'null')))
    dataDictionary['is_crosspostable'].append(submission.setdefault('is_crosspostable', 'false'))
    dataDictionary['is_reddit_media_domain'].append(submission.setdefault('is_reddit_media_domain', 'false'))
    dataDictionary['post_hint'].append(str(submission.setdefault('post_hint', 'null')))
    dataDictionary['num_comments'].append(submission.setdefault('num_comments', 0))
    dataDictionary['permalink'].append(str(submission.setdefault('permalink', 'null')))
    dataDictionary['score'].append(submission.setdefault('score', 0))
    dataDictionary['selftext'].append(str(submission.setdefault('selftext', 'null')))
    dataDictionary['title'].append(str(submission.setdefault('title', 'null')))
    dataDictionary['url'].append(str(submission.setdefault('url', 'null')))
    dataDictionary['comments'].append(str(commentText))
    return True 

# Main Predictor Function
def prediction(url):
    dataDictionary = {'author_fullname': [],
                      'created_utc' : [],
                      'domain' : [],
                      'is_crosspostable' : [],
                      'is_reddit_media_domain' : [],
                      'post_hint' : [],
                      'num_comments' : [],
                      'permalink' : [],
                      'score' : [],
                      'selftext' : [],
                      'title' : [],
                      'url' : [],
                      'comments' : []
                    }

    res = getPostDetails(url,dataDictionary)

    if(res != True):
        return []
    else:
        pandasFrame = pd.DataFrame(dataDictionary)
        pandasFrame['created_utc'] = pandasFrame['created_utc'].apply(classifyTime)
        pandasFrame['domain'] = pandasFrame['domain'].apply(splitUrl)
        pandasFrame['post_hint'] = pandasFrame['post_hint'].apply(cleanText)
        pandasFrame['permalink'] = pandasFrame['permalink'].apply(splitUrl)
        pandasFrame['selftext'] = pandasFrame['selftext'].apply(cleanText)
        pandasFrame['title'] = pandasFrame['title'].apply(cleanText)
        pandasFrame['url'] = pandasFrame['url'].apply(splitUrl)
        pandasFrame['comments'] = pandasFrame['comments'].apply(cleanText)
        pandasFrame = pandasFrame.replace(r'^\s*$', np.nan, regex=True)
        pandasFrame = pandasFrame.replace(np.nan, '')

        newDF = pd.DataFrame()

        for column in pandasFrame.columns:
            if(pandasFrame[column].dtype == 'object'):
                temp = pd.DataFrame(tfidVectorizer[column].transform(pandasFrame[column]).todense(),columns=tfidVectorizer[column].get_feature_names())
                newDF = pd.concat([newDF,temp], axis=1)
                pandasFrame = pandasFrame.drop(columns=column)

        #  Indexing Problem Resolution
        pandasFrame.reset_index(drop=True, inplace=True)
        newDF.reset_index(drop=True, inplace=True)

        pandasFrame = pd.concat([pandasFrame, newDF], axis=1)

        prediction = model.predict_proba(pandasFrame)
        best_3 = np.flip(np.argsort(prediction, axis=1)[:,-3:] ,1)
        best_3 = best_3[0]
        prediction = [labels[x] for x in best_3]
        cssClass = [labeler(x) for x in prediction]
        return([prediction,cssClass])

# Converts labels to CSS Class
def labeler(label):
    try:
        return labelToClass[label]
    except:
        return labelToClass['Food']  
import markdown
import os
import shelve
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import os 


# Import the framework
from flask import Flask, g
from flask_restful import Resource, Api, reqparse

# Create an instance of Flask
app = Flask(__name__)

# Create the API
api = Api(app)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = shelve.open("queries.db")
    return db

@app.teardown_appcontext
def teardown_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
@app.route("/")
def index():
    """Present some documentation"""
    np.random.seed(500)
    file = 'all_dev_with_org.json'
    
    # Open the README file
    #with open(os.path.dirname(app.root_path) + '/README.md', 'r') as markdown_file:
    # Read the content of the file
    content = str("\n ".join(main(file)))

    # Convert to HTML
    return markdown.markdown(content)
def main(file):
    output=[]
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    # converting json dataset from dictionary to dataframe
    train = pd.DataFrame(data)
    train.reset_index(level=0, inplace=True)
    zeroes,ones = train['label'].value_counts()
    output.append('Number of ones: ' + str(ones))
    output.append('Number of zeros : '+ str(zeroes))
    output.append('% of cells labeled ones'+ str(round(ones / len(train) * 100, 2)))
    output.append('% of cells labeled zeros'+ str(round(zeroes / len(train) * 100, 2)))
    # Remove Blank rows in Data, if any
    # Change all the text to lower case
    # Word Tokenization
    # Remove Stop words
    # Remove Non-alpha text
    # Word Lemmatization

    # Step - a : Remove blank rows if any.
    train['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    train['text'] = [entry.lower() for entry in train['text']]
    # Step - c : Tokenization : In this each entry in the train will be broken into set of words
    train['text']= [word_tokenize(entry) for entry in train['text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(train['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        train.loc[index,'text_final'] = str(Final_words)
        #split 
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train['text_final'],train['label'],test_size=0.3)
        Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    # Finally we will transform Train_X and Test_X to vectorized Train_X_Tfidf and Test_X_Tfidf. These will now contain for each row a list of unique integer number and its associated importance as calculated by TF-IDF.

    global Tfidf_vect
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(train['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    global SVM
    SVM = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    output.append("SVM Accuracy Score -> "+ str(accuracy_score(predictions_SVM, Test_Y)*100))
    output.append(classification_report(Test_Y, predictions_SVM))
    return output


def predict (input):
    #input is alist of JSON
    # Remove Blank rows in Data, if any
    # Change all the text to lower case
    # Word Tokenization
    # Remove Stop words
    # Remove Non-alpha text
    # Word Lemmatization

    dataDf=pd.DataFrame([{"text":input}])
    # Step - a : Remove blank rows if any.
    dataDf['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    dataDf['text'] = [entry.lower() for entry in dataDf['text']]
    # Step - c : Tokenization : In this each entry in the dataDf will be broken into set of words
    dataDf['text']= [word_tokenize(entry) for entry in dataDf['text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(dataDf['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        dataDf.loc[index,'text_final'] = str(Final_words)
        xxx = Tfidf_vect.transform(dataDf['text_final'])
        
    return SVM.predict(xxx)



class queryList(Resource):
    def get(self):
        shelf = get_db()
        keys = list(shelf.keys())

        queries = []

        for key in keys:
            queries.append(shelf[key])

        return  queries, 200

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('identifier', required=True)
        parser.add_argument('text', required=True)
        parser.add_argument('output', required=False)
        # Parse the arguments into an object
        args = parser.parse_args()
        output=predict(args['text'])
        shelf = get_db()
        shelf[args['identifier']] = args
        a_str = '\n'.join(str(x) for x in output)
        args["output"]=a_str
        #return {'message': 'Query registered', 'data': args,'output':a_str }, 201
        return args, 201


class Device(Resource):
    def get(self, identifier):
        shelf = get_db()

        # If the key does not exist in the data store, return a 404 error.
        if not (identifier in shelf):
            return {'message': 'Device not found', 'data': {}}, 404

        return {'message': 'Device found', 'data': shelf[identifier]}, 200

    def delete(self, identifier):
        shelf = get_db()

        # If the key does not exist in the data store, return a 404 error.
        if not (identifier in shelf):
            return {'message': 'Device not found', 'data': {}}, 404

        del shelf[identifier]
        return '', 204


api.add_resource(queryList, '/queries')
api.add_resource(Device, '/device/<string:identifier>')





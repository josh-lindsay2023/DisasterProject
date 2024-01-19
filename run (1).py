#import required packages 
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text): #tokenization function same as in train_classifier.py
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
   
    genre_counts = df.groupby('genre').count()['message'] # finds the count of each genre
    genre_names = list(genre_counts.index) # lists genres
    
    Y = df.drop(columns= ["id", "message","genre","original"])#drops non categories columns
    category_names = Y.columns.tolist()# lists categories
    sum_df= pd.DataFrame(Y.sum(), columns= ["sum"])#finds how often each category occurs
    category_counts_by_genre = df.groupby('genre')[category_names].sum().transpose()#finds how often each genre occurs in each category
   
    # create visuals
    
    graphs = [#visual for showing the split among the three genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ),
                
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    graphs.append( # visual for showing the split among each category
    {
        'data': [
            Bar(
                x=category_names,
                y=sum_df["sum"]
            ),
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }
),
    graphs.append(#visual for showing the split amongst each genre acoss each category
    {
        'data': [
            Bar(
                x=category_names,
                y=category_counts_by_genre[genre_name],
                name=genre_name
            )
            for genre_name in genre_names
        ],
        'layout': {
            'title': 'Distribution of Message Categories by Genre',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            },
            'barmode': 'group'  # This will group the bars by category
        }
    }
)
   
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
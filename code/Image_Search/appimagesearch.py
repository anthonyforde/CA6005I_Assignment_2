#===========================================================================================================================
# IMPORT LIBRARIES
#===========================================================================================================================

from flask import Flask, abort, redirect, request, render_template, send_file, send_from_directory, url_for
import os
import math
import numpy as np
import pandas as pd
import csv
import os
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from PIL import Image
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.text import log
import xml.etree.ElementTree as ET

nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

#===========================================================================================================================
# GLOBAL VARIABLES
#===========================================================================================================================

app = Flask(__name__)
df_titles = []
df_Results = []
df_SortedResults = []
df_TopResults = []
vocab = []
vectors = []
search_results = []

#===========================================================================================================================
# FUNCTIONS
#===========================================================================================================================

# ----- FUNCTION: Preprocessing and stopwords removal
# Split document titles into individual words and remove stop words
def preprocess(documents):
    preprocessed_docs = []
    for doc in documents:
        words = doc.lower().split()
        words = [word for word in words if word not in stop_words]
        preprocessed_docs.append(words)
    return preprocessed_docs
# ----- END FUNCTION

# ----- FUNCTION: Vectorisation
# Convert document contents into a vector representation using the vocabulary
def vectorise(doc, vocab):
    vector = np.zeros(len(vocab))
    for word in doc:
        if word in vocab:
            vector[vocab.index(word)] += 1
    return vector
# ----- END FUNCTION

# ----- FUNCTION: Compute cosine similarity between two vectors
def calculate_cosine_similarity(u, v):
    score = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    if np.isnan(score):
      # To cater for values so close to zero they are being treated as NAN
      score = 0
    return score
# ----- END FUNCTION

# ----- FUNCTION: Display image in output
# Split document titles into individual words and remove stop words
def retrieve_image(image_name):

    print("Retrieving image..." + image_name)

    #img_file_path = "image_repository/" + image_name + ".png"

    # Read the image file into a NumPy array
    #img = mpimg.imread(img_file_path)

    # Display the image using matplotlib
    #plt.imshow(img)
    #plt.show()

    # Open the image file
    #img = Image.open(image_name)

    # Display the image
    #img.show()
# ----- END FUNCTION

# ----- FUNCTION: Run search query
def run_query(query):

    global df_titles
    global df_Results
    global df_SortedResults
    global df_TopResults
    global vocab
    global vectors

    df_Results.drop(df_Results.index, inplace=True)

    # Define a regular expression to match unwanted characters
    unwanted_chars_regex = r'[^\w\d\s]'

    # Define a regular expression to match non-ASCII characters
    non_ascii_regex = r'[^\x00-\x7F]'

    # Replace unwanted characters and non-ASCII characters with empty strings
    query = query.replace(unwanted_chars_regex, '').replace(non_ascii_regex, '')

    df_Results.drop(df_Results.index,inplace=True)
    processed_query = query.split()

    for i in range(len(processed_query)):
        processed_query[i] = processed_query[i].lower()

    processed_query = [string for string in processed_query if string not in stop_words]

    query_vec = vectorise(processed_query, vocab)

    # Compute cosine similarity for all documents (previously vectorised above)
    similarities = [calculate_cosine_similarity(query_vec, vector) for vector in vectors]

    current_score = 0
    # For each computed similarity score
    for score in similarities:
        doc_ID = current_score + 1
        # Append a new row to the results dataframe
        #[Doc_ID,	VSM_Score, Rank, Image_Name, Title]

        # Append:
        #new_row = [doc_ID, score, 0, "", ""]
        #df_Results = df_Results.append(pd.Series(new_row, index=df_Results.columns), ignore_index=True)

        # Concat:
        new_row = pd.DataFrame({
        'Doc_ID': [doc_ID],
        'VSM_Score': [score],
        'Rank': [0],
        'Image_Name': [""],
        'Title': [""],
        'Caption': [""]
        })

        df_Results = pd.concat([df_Results, new_row], ignore_index=True)

        current_score += 1

    # Sort and filter results - keep top selection
    df_SortedResults.drop(df_SortedResults.index,inplace=True)
    df_TopResults.drop(df_TopResults.index,inplace=True)
    # Restrict to top 10 results
    num_results_to_return = 10

    df_SortedResults = df_Results.sort_values(by='VSM_Score', ascending=[False])
    # Disregard any results with a similarity of < 0.1
    df_SortedResults = df_SortedResults[df_SortedResults['VSM_Score'] >= 0.1]

    df_TopResults = df_SortedResults.head(num_results_to_return).reset_index(drop=True)
    df_TopResults['Rank'] = range(1, len(df_TopResults) + 1)

    for index, row in df_titles.iterrows():
        df_TopResults.loc[(df_TopResults.Doc_ID == row['Image_ID']), 'Image_Name'] = row['Image_Name']
        df_TopResults.loc[(df_TopResults.Doc_ID == row['Image_ID']), 'Title'] = row['Image_Details']
        df_TopResults.loc[(df_TopResults.Doc_ID == row['Image_ID']), 'Caption'] = row['Caption']

    print("--- QUERY: " + query + "\n")
    print("--- SHOWING TOP RESULTS... "+ "\n\n")

    #df_TopResults
    for index, row in df_TopResults.iterrows():
        print(row['Image_Name'], " | ", row['Caption'], "\n")
        retrieve_image(row['Image_Name'])
        print("\n")
    # ----- END FUNCTION

# ----- FUNCTION: Run search query
def setup():

    global df_titles
    global df_Results
    global df_SortedResults
    global df_TopResults
    global vocab
    global vectors

    indexation_file_path = "/home/afordeprojectspace/.virtualenvs/venv/Image_Search/indexation/Indexed_Images_Preprocessed.csv"

    # Read indexed document titles data into dataframe - title to be used in search results summary
    df_titles = pd.read_csv(indexation_file_path)

    # Create base dataframe for recording results
    df_Results = pd.DataFrame(columns=['Doc_ID', 'VSM_Score', 'Rank', 'Image_Name', 'Title', 'Caption'])
    df_Results.drop(df_Results.index,inplace=True)
    df_SortedResults = df_Results
    df_TopResults = df_Results

    #----------------------------------------------------------------------------------------------------
    # The following code was only run once to create the vocabulary, documents and vectors for the images,
    # and their metadata. The results were exported to pickle files, to be imported and reloaded each
    # time the app is run. There is no need to regenerate these files unless the data changes.
    #----------------------------------------------------------------------------------------------------

    # Import from prepared CSV file - read doc IDs and titles to array
    # with open('Indexed_Images_Preprocessed.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip the first row
    #
    #     documents = []
    #     documentIDs = []
    #     for row in reader:
    #         documentIDs.append(row[0])
    #         documents.append(row[5])
    #
    # # Preprocessing of documents
    # preprocessed_docs = preprocess(documents)
    #
    # # Create vocabulary from the documents
    # vocab = sorted(set(word for doc in preprocessed_docs for word in doc))
    #
    # # Vectorize preprocessed documents
    # vectors = [vectorise(doc, vocab) for doc in preprocessed_docs]
    #
    # with open('preprocessed_docs.pickle', 'wb') as f:
    #     pickle.dump(preprocessed_docs, f)
    #
    # with open('vocab.pickle', 'wb') as f:
    #     pickle.dump(vocab, f)
    #
    # with open('vectors.pickle', 'wb') as f:
    #     pickle.dump(vectors, f)

    # Load the pre-generarted documents and vectors for the image repository

    vocab_file_path = "/home/afordeprojectspace/.virtualenvs/venv/Image_Search/indexation/vocab.pickle"

    with open(vocab_file_path, 'rb') as f:
        vocab = pickle.load(f)

    vectors_file_path = "/home/afordeprojectspace/.virtualenvs/venv/Image_Search/indexation/vectors.pickle"
    with open(vectors_file_path, 'rb') as f:
        vectors = pickle.load(f)
# ----- END FUNCTION

#===========================================================================================================================
# INITIALISATION
#===========================================================================================================================

setup()  # Call the setup function here

#===========================================================================================================================
# FLASK
#===========================================================================================================================

@app.route('/')
def search():
    return render_template('index.html')

# Define the Image_Index page
@app.route('/imageindex')
def imageindex():
    return render_template('imageindex.html')

# Define the samplequeries page
@app.route('/samplequeries')
def samplequeries():
    return render_template('samplequeries.html')

@app.route('/', methods=['POST'])
def results():
    images_folder = os.path.join(app.static_folder, 'images')
    search_query = request.form.get('search')
    print("----------------------------- Looking for image:" + search_query)
    search_results = []

    if search_query:
        print("********** Search input: " + search_query)
        run_query(search_query)

        for index, row in df_TopResults.iterrows():
            search_results.append({
                    'img_name': row['Image_Name'] + ".png",
                    'caption': row['Caption']
                })
            #Test:
            print("********************* Appended to_search results")

    if not search_query or not search_results:
        return render_template('index.html', no_results=True)
    else:
        return render_template('index.html', search_results=search_results, search_query=search_query)

    print("***************** search_results......")
    print(search_results)

    if not search_query or not search_results:
        return render_template('index.html', no_results=True)
    else:
        return render_template('index.html', search_results=search_results, search_query=search_query)
    found_images = []
    for result in search_results:
        img_path = os.path.join(images_folder, result['img_name'])
        if os.path.exists(img_path):
            found_images.append(result)

    # found_images = []
    # for img_name in search_results:
    #     img_path = os.path.join(images_folder, img_name)
    #     if os.path.exists(img_path):
    #         found_images.append(img_name)

    if found_images:
         return render_template('index.html', search_results=found_images, search_query=search_query)
    else:
         return render_template('index.html', no_results=True, search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True)
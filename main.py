# %% [markdown]
# Metadata Based Movie Recommender

# %%
#Import libraries
import pandas as pd
from pathlib import Path

# %%
#Read files

title_path = Path(__file__).parents[0] / 'transformed_data/title.csv'
title = pd.read_csv(title_path)

# %%
#Delete unnamed column
del title['Unnamed: 0']


# %% [markdown]
# Features:
# - cast (top 3)
# - category

# %%
#Parse stringified features into corresponding python objects
from ast import literal_eval

features = ['cast', 'category']
for feature in features:
    title[feature] = title[feature].apply(literal_eval)


# %%
#Function that returns top 3 elements of a list
def get_list(x):
    if isinstance(x, list):
        names = x
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    #Return empty list in case of missing/malformed data
    return []

# %%
#Define new cast and category lists with the get_list(x) function, for the top 3
features = ['cast', 'category']
for feature in features:
    title[feature] = title[feature].apply(get_list)

# %%
#Strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [i.replace(' ', '') for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# %%
#Apply clean_data
features = ['cast', 'category']
for feature in features:
    title[feature] = title[feature].apply(clean_data)

# %%
#Print features
title[['title', 'country', 'cast', 'category', 'user_rating']].head()

# %%
#Create 'metadata soup', a string containing all metadata that will be fed to the vectorizer
def create_soup(x):
    return ' '.join(x['cast']) + ' ' + ' '.join(x['category'])

# %%
#New soup feature
title['soup'] = title.apply(create_soup, axis = 1)

# %%
#Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# %%
#Create the count matrix
count = CountVectorizer(stop_words= 'english')
count_matrix = count.fit_transform(title['soup'])

# %%
#Use cosine_similarity to measure the distance between embeddings
from sklearn.metrics.pairwise import cosine_similarity

# %%
#Compute cosine similarity matrix based on the count_matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# %%
#Reset index of main DataFrame and construct reverse mapping
title = title.reset_index()

# %%
indices = pd.Series(title.index, index = title['title']).drop_duplicates()

# %%
#Define the get_recommendations() function.
def get_recommendations(name, cosine_sim=cosine_sim):
    #Get the index of the movie that matches the title.
    index = indices[name]

    #Get the similarity scores of all movies with that movie.
    sim_scores = list(enumerate(cosine_sim[index]))

    #Sort the movies based on the similarity scores.
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    #Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    #Return the top 10 most similar movies
    result = title['title'].iloc[movie_indices]
    return result

# %% [markdown]
# Streamlit app

# %%
import streamlit as st

# %%
st.title("Basic Movie Recommendation System")
st.image("https://cdn.pixabay.com/photo/2016/03/31/18/36/cinema-1294496__340.png")
st.caption("This app takes the name of a movie and returns ten recommendations based on cast and category.")

name = st.text_input("Movie Title: ")
name = name.lower()

with st.spinner("Getting the recommendations..."):
    try:
        st.write(get_recommendations(name = name, cosine_sim = cosine_sim))
    except KeyError:
        st.write('Please write a movie title.')


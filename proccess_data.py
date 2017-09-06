import pandas as pd
import numpy as np
import pickle

DATA_PATH = 'movie_metadata.csv'
movies = pd.read_csv(DATA_PATH)

# remove non-numeric columns

columns_to_delete = [
    'director_name',
    'actor_1_name',
    'actor_2_name',
    'actor_3_name',
    'genres',
    'language',
    'country',
    'movie_title',
    'movie_imdb_link',
    'plot_keywords'
]

for column in columns_to_delete:
    del movies[column]

labels = movies.columns.tolist()


# drop rows with NAN

movies = movies.dropna()


# map text columns to numbers

color_map = {
    'Color': 1,
    ' Black and White': 0
}

content_rating_map = {
    'G': 0,
    'PG': 1,
    'PG-13': 2,
    'R': 3,
    'TV-14': 4,
    'NC-17': 5,
    'Unrated': 6,
    'Not Rated': 6,
    'Approved': 7,
    'Passed': 8,
    'X': 9,
    'M': 10,
    'GP': 11,
}

movies = movies.replace(to_replace = {
    'color': color_map,
    'content_rating': content_rating_map
})


# move score to end
labels.sort()
score_idx = labels.index('imdb_score')
lables = labels[:score_idx] + labels[score_idx + 1:] + [labels[score_idx]]
movies = movies[labels]


# mean normalization and feature scaling
movies = movies.as_matrix().astype(float)

means = np.mean(movies, axis = 0)
movies = movies - means

stds = np.std(movies, axis = 0)
movies = movies / stds


# save data

A = movies # all data for unsupervised learning
X = movies[:-1] # independent variables for supervised learning
y = movies[-1] # dependent variable for supervised learning

f = open('movie_data.p', 'wb')
pickle.dump({
    'labels': labels,
    'A': A,
    'X': X,
    'y': y
}, f)
f.close()

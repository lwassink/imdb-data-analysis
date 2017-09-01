import pandas as pd

DATA_PATH = 'movie_metadata.csv'

movies = pd.DataFrame.from_csv(DATA_PATH)

# remove non-numeric columns

columns_to_delete = [
    'director_name',
    'actor_1_name',
    'actor_2_name',
    'actor_3_name',
    'genres',
    'language',
    'country',
]

for column in columns_to_delete:
    del movies[column]

print(movies.columns)

     #  'content_rating' , 'color'

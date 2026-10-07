import numpy as np
import pandas as pd
import ast
import pickle
import request
import streamlit as st


# 1. LOAD DATA


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(movies.head())
print(credits.head())


# 2. MERGE MOVIES + CREDITS


movies = movies.merge(credits, on="title")

print(movies.head())
print(movies.shape)

# 3. SELECT REQUIRED COLUMNS

movies = movies[
    ["id", "title", "overview", "genres", "keywords", "cast", "crew"]
]

print(movies.head())

# 4. MISSING VALUES


print("Missing values:")
print(movies.isnull().sum())

movies.dropna(inplace=True)

# IMPORTANT:
# Reset index after dropping rows
movies.reset_index(drop=True, inplace=True)


# 5. DUPLICATE CHECk

print("Duplicate rows:", movies.duplicated().sum())

movies.drop_duplicates(inplace=True)

# Reset index again
movies.reset_index(drop=True, inplace=True)

# 6. CONVERT GENRES / KEYWORDS

def convert(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i["name"])

    return L


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

# 7. CONVERT CAST


def convert3(text):
    L = []

    counter = 0

    for i in ast.literal_eval(text):

        if counter < 3:
            L.append(i["name"])

        counter += 1

    return L


movies["cast"] = movies["cast"].apply(convert3)

# 8. EXTRACT DIRECTOR

def fetch_director(text):

    L = []

    for i in ast.literal_eval(text):

        if i["job"] == "Director":
            L.append(i["name"])

    return L


movies["crew"] = movies["crew"].apply(fetch_director)

# 9. CONVERT OVERVIEW INTO LIST

movies["overview"] = movies["overview"].apply(lambda x: x.split())

# 10. REMOVE SPACES FROM WORDS

def collapse(L):

    L1 = []

    for i in L:
        L1.append(i.replace(" ", ""))

    return L1


movies["genres"] = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"] = movies["cast"].apply(collapse)
movies["crew"] = movies["crew"].apply(collapse)

print(movies.head())

# 11. CREATE TAGS

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

# 12. CREATE FINAL DATAFRAME


new = movies[
    ["id", "title", "tags"]
].copy()

new["tags"] = new["tags"].apply(lambda x: " ".join(x))

print(new.head())

# 13. COUNT VECTORIZER

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vector = cv.fit_transform(new["tags"]).toarray()

print("Vector shape:", vector.shape)

# 14. COSINE SIMILARITY


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

print("Similarity shape:", similarity.shape)


# 15. RECOMMENDATION FUNCTION


def recommend(movie):

    if movie not in new["title"].values:
        return []

    index = new[new["title"] == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []

    for i in distances[1:6]:

        recommended_movies.append(
            new.iloc[i[0]]["title"]
        )

    return recommended_movies


# Test
print(recommend("Gandhi"))

# 16. SAVE MODEL


pickle.dump(new, open("movie_list.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Model files saved successfully.")

# 17. STREAMLIT APP


st.header("🎬 Movie Recommender System")

# Load saved files
movie_data = pickle.load(
    open("movie_list.pkl", "rb")
)

similarity_data = pickle.load(
    open("similarity.pkl", "rb")
)

movie_list = movie_data["title"].values

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# 18. FETCH POSTER


def fetch_poster(movie_id):

    # Put your TMDB API key in Streamlit secrets
    api_key = st.secrets["TMDB_API_KEY"]

    url = (
        f"https://api.themoviedb.org/3/movie/"
        f"{movie_id}?api_key={api_key}&language=en-US"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    poster_path = data.get("poster_path")

    if poster_path is None:
        return None

    return (
        "https://image.tmdb.org/t/p/w500/"
        + poster_path
    )

# 19. FINAL RECOMMENDATION FUNCTION

def get_recommendations(movie):

    index = movie_data[
        movie_data["title"] == movie
    ].index[0]

    distances = sorted(
        list(enumerate(similarity_data[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:

        movie_row = movie_data.iloc[i[0]]

        movie_id = movie_row["id"]

        poster = fetch_poster(movie_id)

        recommended_movie_names.append(
            movie_row["title"]
        )

        recommended_movie_posters.append(
            poster
        )

    return (
        recommended_movie_names,
        recommended_movie_posters
    )



# 20. SHOW RECOMMENDATIONS


if st.button("Show Recommendation"):

    names, posters = get_recommendations(
        selected_movie
    )

    cols = st.columns(5)

    for col, name, poster in zip(
        cols,
        names,
        posters
    ):

        with col:

            st.text(name)

            if poster:
                st.image(poster)
            else:
                st.write("Poster not available")
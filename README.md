 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python and Machine Learning.

 Project Overview

This project recommends movies based on the similarity between movies.

The system analyzes movie information such as:
- Genres
- Keywords
- Overview
- Cast
- Crew

It then finds movies that are most similar to the movie selected by the user.

 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- TMDB 5000 Movies Dataset

 How It Works

1. Load the TMDB movie dataset
2. Clean and preprocess the data
3. Combine important movie features
4. Convert text features into numerical vectors
5. Calculate similarity between movies
6. Recommend similar movies

 Features

- Search for a movie
- Get similar movie recommendations
- Content-based recommendation
- Simple Streamlit interface

 Project Structure

```text
movie-recommendation-system/
│
├── movie_recommendation.py
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── README.md

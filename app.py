import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the song dataset
@st.cache_data
def load_song_data():
    file_path = 'song_data.csv'
    song_data = pd.read_csv(file_path)
    
    np.random.seed(42)
    moods = ['happy', 'sad', 'energetic', 'calm']
    song_data['mood'] = np.random.choice(moods, size=len(song_data))

    users = [f'user_{i}' for i in range(1, 101)]
    user_song_interactions = pd.DataFrame({
        'user_id': np.random.choice(users, size=500),
        'song_id': np.random.choice(song_data['song_id'], size=500),
        'rating': np.random.randint(1, 6, size=500)  # Ratings between 1 and 5
    })
    
    return song_data, user_song_interactions

# Load the novels dataset
file_path = 'novel_data.csv'
novels_data = pd.read_csv(file_path)

# Add mock mood labels to the novels dataset
np.random.seed(42)
moods = ['happy', 'sad', 'energetic', 'calm']
novels_data['Mood'] = np.random.choice(moods, size=len(novels_data))

# Feature selection: Select relevant features for mood prediction
features_novels = ['User Rating', 'Reviews', 'Price', 'Year']

# Handle missing values using SimpleImputer (example with mean)
imputer_novels = SimpleImputer(strategy='mean')
X_imputed_novels = imputer_novels.fit_transform(novels_data[features_novels])
y_novels = novels_data['Mood']

# Split data into training and testing sets
X_train_novels, X_test_novels, y_train_novels, y_test_novels = train_test_split(X_imputed_novels, y_novels, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier with optimized parameters
clf_novels = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier on the training data
clf_novels.fit(X_train_novels, y_train_novels)

# Load the movie dataset
@st.cache_data
def load_movie_data():
    file_path = 'movie_data.csv'
    movie_data = pd.read_csv(file_path)
    
    np.random.seed(42)
    moods = ['happy', 'sad', 'energetic', 'calm']
    movie_data['Mood'] = np.random.choice(moods, size=len(movie_data))

    features_movies = ['RottenTomatoes', 'AudienceScore', 'TheatersOpenWeek', 'OpeningWeekend', 
                       'DomesticGross', 'ForeignGross', 'Budget']
    imputer_movies = SimpleImputer(strategy='mean')
    X_imputed_movies = imputer_movies.fit_transform(movie_data[features_movies])
    y_movies = movie_data['Mood']
    
    X_train_movies, X_test_movies, y_train_movies, y_test_movies = train_test_split(X_imputed_movies, y_movies, test_size=0.2, random_state=42)
    clf_movies = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_movies.fit(X_train_movies, y_train_movies)
    
    return movie_data, clf_movies, imputer_movies, features_movies

song_data, user_song_interactions = load_song_data()
movie_data, clf_movies, imputer_movies, features_movies = load_movie_data()

# Create a utility matrix for songs
user_song_matrix = user_song_interactions.pivot(index='user_id', columns='song_id', values='rating').fillna(0)
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_song_matrix)
song_factors = svd.components_
predicted_ratings = np.dot(user_factors, song_factors)

# Song recommendation function
def recommend_songs(mood, num_recommendations=5):
    average_ratings = predicted_ratings.mean(axis=0)
    mood_songs = song_data[song_data['mood'] == mood]['song_id']
    mood_song_indices = [user_song_matrix.columns.get_loc(song) for song in mood_songs if song in user_song_matrix.columns]
    mood_ratings = average_ratings[mood_song_indices]
    top_indices = mood_ratings.argsort()[::-1][:num_recommendations]
    recommended_songs = [user_song_matrix.columns[mood_song_indices[i]] for i in top_indices]
    recommended_songs_df = song_data[song_data['song_id'].isin(recommended_songs)]
    return recommended_songs_df

# Function to recommend novels based on user's mood input
def recommend_novels_by_mood(mood):
    predicted_moods = clf_novels.predict(X_imputed_novels)
    predicted_novels = novels_data[predicted_moods == mood]
    return predicted_novels[['Name', 'Author', 'Reviews', 'Year']]

# Movie recommendation function
def recommend_movies_by_mood(mood):
    X_imputed = imputer_movies.transform(movie_data[features_movies])
    predicted_moods = clf_movies.predict(X_imputed)
    recommended_movies = movie_data[predicted_moods == mood]
    return recommended_movies[['Movie', 'LeadStudio', 'Genre', 'Year']]

# Apply custom CSS
st.markdown("""
    <style>
        /* General page background */
        body {
            background-color: white;
        }

        /* Main content background */
        .stApp {
            background-color: white;
        }

        /* Selectbox label color */
        .css-1cpxqw2.e1fqkh3o3 {
            color: #00AFEF;
        }

         /* Button styles */
.stButton button {
    background-color: #00AFEF; /* Blue background */
    color: white;
    font-size: var(--fs-11);
    font-weight: var(--fw-700);
    text-transform: uppercase;
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 30px;
    border: 2px solid #00AFEF; /* Blue border */
    border-radius: 50px;
    transition: background-color 0.3s, border-color 0.3s;
}

.stButton button:hover,
.stButton button:focus {
    background-color: rgb(235, 46, 188); /* Pink background */
    border-color: rgb(235, 46, 188); /* Pink border */
    color: white;
}



/* Ensure the button does not change color after being clicked */
.stButton button:visited,
.stButton button:active:focus {
    background-color: #00AFEF; /* Blue background */
    border-color: #00AFEF; /* Blue border */
    color: white;
}


        /* Header styles */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #00AFEF;
        }

        /* Table row background */
        .stDataFrame tr:nth-child(even) {
            background-color: white;
        }
        .stDataFrame tr:nth-child(odd) {
            background-color: white;
        }

        /* Table text color */
        .stDataFrame td, .stDataFrame th {
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title('Recommendations')

# User selection for category
category = st.selectbox('Select Category:', ['Songs', 'Movies', 'Novels'])

# User Interface for selecting mood
if category == 'Songs':
    moods = song_data['mood'].unique()
    selected_mood = st.selectbox('Select Mood:', moods)
    if st.button('Get Song Recommendations'):
        recommendations = recommend_songs(selected_mood)
        st.write('Recommended Songs:')
        st.write(recommendations)
    
    # Evaluation for songs
    st.header('Model Evaluation (Songs)')
    actual = user_song_matrix.values.flatten()
    predicted = predicted_ratings.flatten()
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)


elif category == 'Movies':
    moods = movie_data['Mood'].unique()
    selected_mood = st.selectbox('Select Mood:', moods)
    if st.button('Get Movie Recommendations'):
        recommendations = recommend_movies_by_mood(selected_mood)
        st.write('Recommended Movies:')
        st.write(recommendations)

elif category == 'Novels':
    moods = novels_data['Mood'].unique()
    selected_mood = st.selectbox('Select Mood:', moods)
    if st.button('Get Novel Recommendations'):
        recommendations = recommend_novels_by_mood(selected_mood)
        st.write('Recommended Novels:')
        st.write(recommendations)

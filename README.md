# Movie Recommender System

A content-based movie recommendation system built with Streamlit that uses machine learning to suggest movies based on similarity analysis of genres, keywords, cast, crew, and plot overviews.

## Features

- **Content-Based Recommendations**: Analyzes movie metadata (genres, keywords, cast, director, overview) to find similar movies
- **Web Interface**: User-friendly Streamlit application for easy interaction
- **Poster Display**: Fetches and displays movie posters from TMDB API with fallback placeholder images
- **Caching**: Efficient caching mechanisms for data loading and model building to improve performance
- **Retry Logic**: Robust HTTP session with automatic retry logic for API calls
- **Deployable**: Includes Procfile for easy deployment to Heroku or similar platforms

## How It Works

1. **Data Processing**: Loads TMDB movie and credits datasets
2. **Feature Engineering**: Extracts and combines:
   - Movie genres
   - Keywords
   - Cast (top 3 actors)
   - Director
   - Plot overview
3. **Text Processing**: Uses Porter Stemmer for text normalization
4. **Similarity Calculation**: Applies CountVectorizer and cosine similarity to build a recommendation model
5. **Caching**: Stores pre-computed model in `model_cache.pkl` for faster subsequent runs

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- NLTK
- Requests


## Setup

### Environment Variables

Set the following environment variable before running the application:

```bash
export TMDB_API_KEY="your_tmdb_api_key_here"
```

Get your API key from [The Movie Database (TMDB)](https://www.themoviedb.org/settings/api)

### Data Files

Ensure the following CSV files are in the project root:

- `tmdb_5000_movies.csv` - Movie metadata
- `tmdb_5000_credits.csv` - Cast and crew information

## Running Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your TMDB API key:

   ```bash
   export TMDB_API_KEY="your_api_key"
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

The application will open in your browser at `http://localhost:8501`


## Project Structure

```
Movie_Recommender_System/
├── main.py                      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── setup.sh                     # setup script
├── Procfile                     # configuration
├── tmdb_5000_movies.csv         # Movie metadata dataset
├── tmdb_5000_credits.csv        # Cast and crew dataset
└── README.md                    # This file
```

## Technical Details

### Caching Strategy

- **Data Loading**: Cached with Streamlit's `@st.cache_data` to load CSV files only once per session
- **Model Building**: Pre-computed model cached in `model_cache.pkl` for immediate load times on subsequent runs
- **Poster Fetching**: 24-hour TTL cache for TMDB API poster requests


## Future Enhancements

- Collaborative filtering recommendations
- User ratings and feedback integration
- Advanced filtering options (year, rating, runtime)
- Search and filter by actor/director
- Recommendation explanation features

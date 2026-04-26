import numpy as np
import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Cache-friendly HTTP session with retry logic ──────────────────────────────
@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
PLACEHOLDER   = "https://via.placeholder.com/500x750?text=No+Poster"

# ── Data loading & model building (cached so it only runs once per session) ───

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    movies  = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    return movies.merge(credits, on="title")

def convert(obj):
    return [i["name"] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i["name"] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []

def stem(text, ps=PorterStemmer()):
    return " ".join(ps.stem(w) for w in text.split())

@st.cache_data(show_spinner="Building recommendation model…")
def build_model():
    # ── Try to load pre-computed artefacts first ──────────────────────────────
    if os.path.exists("model_cache.pkl"):
        with open("model_cache.pkl", "rb") as f:
            return pickle.load(f)

    df = load_data()
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    df.dropna(inplace=True)

    df["genres"]   = df["genres"].apply(convert)
    df["keywords"] = df["keywords"].apply(convert)
    df["cast"]     = df["cast"].apply(convert3)
    df["crew"]     = df["crew"].apply(fetch_director)
    df["overview"] = df["overview"].apply(lambda x: x.split())

    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(lambda lst: [i.replace(" ", "") for i in lst])

    df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]

    new_df = df[["movie_id", "title"]].copy()
    new_df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv      = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    sim     = cosine_similarity(vectors)

    result = {"new_df": new_df, "similarity": sim}
    with open("model_cache.pkl", "wb") as f:
        pickle.dump(result, f)

    return result

# ── Poster fetching (cached per movie_id, parallel-friendly) ──────────────────

@st.cache_data(ttl=86400, show_spinner=False)   # cache for 24 h
def fetch_poster(movie_id: int) -> str:
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    try:
        resp = get_session().get(url, timeout=5)
        resp.raise_for_status()
        path = resp.json().get("poster_path")
        if path:
            return f"https://image.tmdb.org/t/p/w500{path}"
    except requests.RequestException:
        pass
    return PLACEHOLDER

# ── Recommendation logic ──────────────────────────────────────────────────────

def recommend(movie: str, model: dict):
    new_df     = model["new_df"]
    similarity = model["similarity"]

    idx       = new_df[new_df["title"] == movie].index[0]
    distances = similarity[idx]
    top5      = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]

    names, posters = [], []
    for i, _ in top5:
        row = new_df.iloc[i]
        names.append(row.title)
        posters.append(fetch_poster(int(row.movie_id)))   # each call is cached

    return names, posters

# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.title("Movie Recommender System")

    model  = build_model()
    new_df = model["new_df"]

    selected = st.selectbox(
        "Select a movie from the dropdown",
        new_df["title"].values,
    )

    if st.button("Recommend"):
        with st.spinner("Fetching recommendations…"):
            names, posters = recommend(selected, model)

        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.text(name)
                st.image(poster)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import random
from pathlib import Path
import pickle

from recwalk import RecWalk

@st.cache_data(show_spinner=False)
def load_artist_mapping(data_path: Path):
    try:
        artists = pd.read_csv(data_path / "artists.dat", sep="\t", encoding="utf-8")
    except UnicodeDecodeError:
        artists = pd.read_csv(data_path / "artists.dat", sep="\t", encoding="utf-8", errors="replace")

    return dict(zip(artists.id, artists.name))

@st.cache_data(show_spinner=False)
def load_users(data_path: Path):
    user_ids = list(range(1, 101))
    return user_ids

@st.cache_data(show_spinner=False)
def get_recwalk(data_path: Path):
    artists = load_artist_mapping(data_path)
    tags = pd.read_csv("lastfm/tags.dat", sep="\t", encoding="latin1")
    user_artists = pd.read_csv("lastfm/user_artists.dat", sep="\t", encoding="latin1")
    user_tags = pd.read_csv("lastfm/user_taggedartists.dat", sep="\t", encoding="latin1")
    user_friends = pd.read_csv("lastfm/user_friends.dat", sep="\t", encoding="latin1")
    recwalk = RecWalk(alpha=0.5)

    with open("models/recwalk/entity2id_enhanced.pkl", "rb") as f:
        entity2id = pickle.load(f)

    recwalk.build_graph(user_artists, user_friends, user_tags, entity2id)
    return recwalk

def main():
    st.set_page_config(page_title="Last.fm Recommender", page_icon="🎧", layout="wide")
    st.title("🎧 Last.fm Recommender Demo")

    data_path = Path("lastfm")
    image_dir = Path("images")
    default_img = image_dir / "default.jpg"

    # Load data
    artist_map = load_artist_mapping(data_path)
    artist_ids = list(artist_map.keys())
    user_ids = load_users(data_path)
    recwalk = get_recwalk(data_path)

    # Sidebar: Select model and user
    st.sidebar.header("Recommender Settings")
    selected_model = st.sidebar.selectbox("Choose Model", ["Random", "RecWalk", "Dynamic_user_weights"])
    selected_user = st.sidebar.selectbox("Choose User", user_ids)
    top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    st.markdown(f"**Model:** {selected_model} | **User:** {selected_user}")
    if st.button("Recommend", use_container_width=True):
        with st.spinner("Generating recommendations…"):
            # Placeholder logic: currently still random
            if selected_model == "Random":
                rec_ids = random.sample(artist_ids, min(top_k, len(artist_ids)))
            elif selected_model == "RecWalk":
                recommended = recwalk.recommend(selected_user, top_k=top_k)
                rec_ids = [a for a, _ in recommended]
            else:
                rec_ids = random.sample(artist_ids, min(top_k, len(artist_ids)))

            rec_artists = [(a_id, artist_map[a_id]) for a_id in rec_ids]

            cols = st.columns(5)
            for i, (artist_id, artist_name) in enumerate(rec_artists):
                img_path = image_dir / f"{artist_id}.jpg"
                display_img = img_path if img_path.exists() else default_img
                with cols[i % 5]:
                    st.image(str(display_img), caption=artist_name, use_container_width=True)



if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import random
from pathlib import Path
import pickle

from recwalk import RecWalk
from hybrid_recommender import HybridRecommender

@st.cache_data(show_spinner=False)
def load_artist_mapping(data_path: Path):
    try:
        artists = pd.read_csv(data_path / "artists.dat", sep="\t", encoding="utf-8")
    except UnicodeDecodeError:
        artists = pd.read_csv(data_path / "artists.dat", sep="\t", encoding="utf-8", errors="replace")

    return dict(zip(artists.id, artists.name))

@st.cache_data(show_spinner=False)
def load_users(user_artists):
    user_ids = sorted(set(user_artists.userID))
    return user_ids

@st.cache_data(show_spinner=False)
def load_data():
    tags = pd.read_csv("lastfm/tags.dat", sep="\t", encoding="latin1")
    user_artists = pd.read_csv("lastfm/user_artists.dat", sep="\t", encoding="latin1")
    user_tags = pd.read_csv("lastfm/user_taggedartists.dat", sep="\t", encoding="latin1")
    user_friends = pd.read_csv("lastfm/user_friends.dat", sep="\t", encoding="latin1")
    return tags, user_artists, user_tags, user_friends

@st.cache_data(show_spinner=False)
def get_recwalk(user_artists, user_friends, user_tags, entity2id):
    
    recwalk = RecWalk(alpha=0.5)

    recwalk.build_graph(user_artists, user_friends, user_tags, entity2id)
    return recwalk

def get_artist_ids(artists):
    ids = []
    for artist in artists:
        id = int(artist.replace('artist_', ''))
        ids.append(id)
    return ids

def main():
    st.set_page_config(page_title="Last.fm Recommender", page_icon="🎧", layout="wide")
    st.title("🎧 Last.fm Recommender Demo")

    data_path = Path("lastfm")
    image_dir = Path("images")
    default_img = image_dir / "default.jpg"

    with open("models/recwalk/entity2id_enhanced.pkl", "rb") as f:
        entity2id = pickle.load(f)

    # Load data
    artists = load_artist_mapping(data_path)
    tags, user_artists, user_tags, user_friends = load_data()

    artist_map = load_artist_mapping(data_path)
    artist_ids = list(artist_map.keys())
    user_ids = load_users(user_artists)
    recwalk = get_recwalk(user_artists, user_friends, user_tags, entity2id)
    hybrid = HybridRecommender(model_dir="models/hybrid", k=30)

    # Sidebar: Select model and user
    st.sidebar.header("Recommender Settings")
    selected_model = st.sidebar.selectbox("Choose Model", ["Random", "RecWalk", "Hybrid_Common", "Hybrid_Personalised"])
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
            elif selected_model == "Hybrid_Common":
                user_name = f"user_{selected_user}"
                recs_common = hybrid.recommend(user_name, user_friends, user_tags, personalized=False)
                rec_ids = get_artist_ids(recs_common)
                rec_ids = [a for a in rec_ids if a in artist_ids][:top_k]
            elif selected_model == "Hybrid_Personalised":
                user_name = f"user_{selected_user}"
                recs_common = hybrid.recommend(user_name, user_friends, user_tags, personalized=True)
                rec_ids = get_artist_ids(recs_common)
                rec_ids = [a for a in rec_ids if a in artist_ids][:top_k]
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

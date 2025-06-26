import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import random
from pathlib import Path
import pickle
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from recwalk import SimilarityEnhancedRecWalk
from hybrid_recommender import HybridRecommender
from ease import EASE

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
def load_user_history(user_artists):
    # Create a dictionary of userID to their listened artists
    return user_artists.groupby('userID')['artistID'].apply(list).to_dict()

@st.cache_data(show_spinner=False)
def load_data():
    tags = pd.read_csv("lastfm/tags.dat", sep="\t", encoding="latin1")
    user_artists = pd.read_csv("lastfm/user_artists.dat", sep="\t", encoding="latin1")
    user_tags = pd.read_csv("lastfm/user_taggedartists.dat", sep="\t", encoding="latin1")
    user_friends = pd.read_csv("lastfm/user_friends.dat", sep="\t", encoding="latin1")
    artist_genres = pd.read_csv("lastfm/artist_genres.csv", encoding="latin1")

    # Filter users with sufficient interactions (at least 2)
    user_counts = user_artists['userID'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    filtered_data = user_artists[user_artists['userID'].isin(valid_users)]
    
    # Split into train/test (stratified by user)
    train_data, test_data = train_test_split(
        filtered_data,
        test_size=0.2,
        random_state=1023,
        stratify=filtered_data['userID']
    )
    return tags, user_artists, user_tags, user_friends, artist_genres, train_data, test_data

@st.cache_data(show_spinner=False)
def get_recwalk(user_artists, user_friends, user_tags, artist_genres, entity2id):
    model = SimilarityEnhancedRecWalk(alpha=0.1, genre_weight=0.5, sim_weight=0.7)
    model.build_graph(user_artists, user_friends, user_tags, artist_genres, entity2id)
    return model

@st.cache_data(show_spinner=False)
def train_ease(train_data):
    ease = EASE(reg_weight=1.0)
    ease.fit(train_data)
    return ease

@st.cache_data(show_spinner=False)
def get_artist_ids(artists):
    ids = []
    for artist in artists:
        id = int(artist.replace('artist_', ''))
        ids.append(id)
    return ids

def display_artists(artist_list, ground_truth=None, columns=5):
    """Helper function to display artists in a grid"""
    default_img = Path("images/default.jpg")
    cols = st.columns(columns)
    for i, (artist_id, artist_name) in enumerate(artist_list):
        img_path = Path(f"images/{artist_id}.jpg")
        display_img = img_path if img_path.exists() else default_img
        with cols[i % columns]:
            # Highlight matches if ground truth is provided
            if ground_truth and artist_id in ground_truth:
                st.success(f"✅ {artist_name}")
            else:
                st.image(str(display_img), caption=artist_name, use_container_width=True)

def calculate_ndcg(ground_truth, recommendations, top_k):
    """Calculate Normalized Discounted Cumulative Gain"""
    # Create relevance scores (1 for relevant, 0 otherwise)
    relevance = [1 if artist_id in ground_truth else 0 for artist_id in recommendations[:top_k]]
    
    # Calculate DCG
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    
    # Calculate Ideal DCG
    ideal_relevance = [1] * min(len(ground_truth), top_k)
    idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    
    return dcg / idcg if idcg > 0 else 0

def visualize_paths_flowchart(paths):
    """Display recommendation paths in a clean flowchart format with emojis"""
    if not paths:
        return "No connecting paths found"
    
    visualization = []
    path_icons = {
        'direct': '🔗',
        'friend': '👥',
        'genre': '🏷️',
        'similarity': '🔄'
    }
    
    for path in paths:
        viz = []
        path_type = path['path_type']
        weight = path['weight']
        details = path['details']
        
        # Header with emoji
        viz.append(f"╭── {path_icons.get(path_type, '')} {path_type.upper()} PATH ──╮")
        viz.append(f"│ Strength: {weight:.4f}")
        viz.append("├──────────────────────")
        
        # Path visualization with emojis
        if path_type == 'direct':
            viz.append("│ 👤 YOU → 🎵 Listened → 🎤 ARTIST")
            viz.append(f"│ Direct weight: {details['direct_weight']:.4f}")
            
        elif path_type == 'friend':
            viz.append(f"│ 👤 YOU → 👥 Friends With → 👤 USER {details['friend_id']}")
            viz.append(f"│            → 🎵 Listened → 🎤 ARTIST")
            viz.append(f"│ Friend connection: {details['user_friend_weight']:.4f}")
            viz.append(f"│ Their listening: {details['friend_artist_weight']:.4f}")
            
        elif path_type == 'genre':
            genre = details['genre'].replace('genre_', '')[:20]
            viz.append(f"│ 👤 YOU → ❤️ Like → 🏷️ {genre}")
            viz.append(f"│            → 🔗 Describes → 🎤 ARTIST")
            viz.append(f"│ Genre matches: {details['user_genre_count']}")
            
        elif path_type == 'similarity':
            via_name = artist_map.get(int(details['via_artist'].replace('artist_', '')), 
                                    details['via_artist'])
            viz.append(f"│ 👤 YOU → 🎵 Listened → 🎤 {via_name[:15]}")
            viz.append(f"│            → 🔄 Similar → 🎤 ARTIST")
            viz.append(f"│ Similarity: {details['similarity_score']:.4f}")
            viz.append(f"│ Your listening: {details['user_artist_weight']:.4f}")
        
        # Footer
        viz.append("╰──────────────────────╯")
        visualization.append("\n".join(viz))
    
    return "\n\n".join(visualization)

def main():
    st.set_page_config(page_title="Last.fm Recommender", page_icon="🎧", layout="wide")
    st.title("🎧 Last.fm Recommender Demo")

    data_path = Path("lastfm")
    image_dir = Path("images")
    default_img = image_dir / "default.jpg"

    with open("models/enhanced_recwalk/entity2id.pkl", "rb") as f:
        entity2id = pickle.load(f)

    # Load data
    tags, user_artists, user_tags, user_friends, artist_genres, train_data, test_data = load_data()
    artist_map = load_artist_mapping(data_path)
    artist_ids = list(artist_map.keys())
    # user_ids = load_users(user_artists)
    user_ids = load_users(test_data)
    # user_history = load_user_history(user_artists)
    user_history = load_user_history(train_data)
    # recwalk = get_recwalk(user_artists, user_friends, user_tags, artist_genres, entity2id)
    recwalk = get_recwalk(train_data, user_friends, user_tags, artist_genres, entity2id)
    hybrid = HybridRecommender(model_dir="models/hybrid", k=30)
    ease = train_ease(train_data)

    # Sidebar: Select model and user
    st.sidebar.header("Recommender Settings")
    selected_model = st.sidebar.selectbox("Choose Model", ["RecWalk", "EASE", "Hybrid", "Random"])
    selected_user = st.sidebar.selectbox("Choose User", user_ids)
    top_k = st.sidebar.slider("Number of Recommendations", 5, 50, 50)
    show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=True)

    st.markdown(f"**Model:** {selected_model} | **User:** {selected_user}")
    
    if st.button("Generate Recommendations", use_container_width=True, type="primary"):
        with st.spinner("Generating recommendations..."):
            # Get ground truth for selected user
            ground_truth_ids = user_history.get(selected_user, [])
            ground_truth_artists = [(a_id, artist_map.get(a_id, f"Unknown Artist ({a_id})")) 
                                  for a_id in ground_truth_ids]
            
            # Generate recommendations
            if selected_model == "Random":
                rec_ids = random.sample(artist_ids, min(top_k, len(artist_ids)))
            elif selected_model == "RecWalk":
                user_id = user_ids.index(selected_user)
                recommended = recwalk.recommend(selected_user, top_k=top_k)
                rec_ids = [a for a, _ in recommended]
            elif selected_model == "EASE":
                recommended = ease.recommend(selected_user, train_data, top_k=top_k)
                rec_ids = [a for a, _ in recommended]
            elif selected_model == "Hybrid":
                user_name = f"user_{selected_user}"
                recs_common = hybrid.recommend(user_name, user_friends, user_tags, personalized=False)
                rec_ids = get_artist_ids(recs_common)
                rec_ids = [a for a in rec_ids if a in artist_ids][:top_k]
            else:
                rec_ids = random.sample(artist_ids, min(top_k, len(artist_ids)))

            rec_artists = [(a_id, artist_map.get(a_id, f"Unknown Artist ({a_id})")) 
                         for a_id in rec_ids]

            # Display ground truth
            with st.expander("🎵 User's Listening History (Ground Truth)", expanded=False):
                display_artists(ground_truth_artists, columns=5)
                st.caption(f"Showing {len(ground_truth_artists)} artists from user's history")

            st.divider()

            # Display recommendations
            st.subheader(f"🎧 {selected_model} Recommendations")
            display_artists(rec_artists, set(ground_truth_ids), columns=5)

            if selected_model == "RecWalk" and rec_ids:
                top_artists = rec_ids
                
                for i, artist_id in enumerate(top_artists, 1):
                    artist_name = artist_map.get(artist_id, f"Artist {artist_id}")
                    
                    with st.expander(f"🎤 Recommendation #{i}: {artist_name}", expanded=(i==1)):
                        paths = recwalk.get_kg_paths(user_id=selected_user, artist_id=artist_id)
                        
                        if paths:
                            # Create columns for better layout
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                # Artist image
                                img_path = Path(f"images/{artist_id}.jpg")
                                display_img = img_path if img_path.exists() else default_img
                                st.image(str(display_img), width=150)
                                
                            with col2:
                                # Path visualization
                                st.markdown("#### Connection Paths")
                                st.code(visualize_paths_flowchart(paths), language='text')
                                
                                # Show metrics
                                total_weight = sum(p['weight'] for p in paths)
                                st.caption(f"Total connection strength: {total_weight:.3f}")
                        else:
                            st.warning("No connecting paths found for this recommendation")
                            
                        st.divider()
            
            # Show evaluation metrics if enabled
            if show_metrics and ground_truth_ids:
                overlap = len(set(ground_truth_ids) & set(rec_ids))
                precision = overlap / top_k if top_k > 0 else 0
                recall = overlap / len(ground_truth_ids) if ground_truth_ids else 0
                ndcg = calculate_ndcg(set(ground_truth_ids), rec_ids, top_k)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Matching Artists", overlap)
                with col2:
                    st.metric("Precision", f"{precision:.1%}")
                with col3:
                    st.metric("Recall", f"{recall:.1%}")
                with col4:
                    st.metric("NDCG", f"{ndcg:.3f}")
                    

if __name__ == "__main__":
    main()
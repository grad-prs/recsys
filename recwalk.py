import scipy.sparse as sp
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import ParameterGrid
import scipy.sparse as sp
from sklearn.preprocessing import normalize

class RecWalk:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.user2idx = {}  # Maps userID to graph index
        self.artist2idx = {}  # Maps artistID to graph index
        self.tag2idx = {}  # Maps tagID to graph index
        self.P = None  # Transition probability matrix
        self.popular_artists = None  # Cache for popular artists fallback

    def build_graph(self, user_artists, user_friends, user_tags, entity2id):
        num_entities = len(entity2id)
        adj = sp.lil_matrix((num_entities, num_entities))
        
        # Initialize all mappings
        self.user2idx = {}
        self.artist2idx = {}
        self.tag2idx = {}
        
        # 1. Process user-artist interactions (training data only)
        artist_weights = defaultdict(float)
        for _, row in user_artists.iterrows():
            user_id = f"user_{row['userID']}"
            artist_id = f"artist_{row['artistID']}"
            if user_id in entity2id and artist_id in entity2id:
                # Store mappings
                self.user2idx[row['userID']] = entity2id[user_id]
                self.artist2idx[row['artistID']] = entity2id[artist_id]
                
                # Weight edges by play count (log-normalized)
                # weight = 3 / np.log1p(row.get('weight', 1))
                weight = np.log1p(row.get('weight', 1)) 
                u_idx = entity2id[user_id]
                a_idx = entity2id[artist_id]
                adj[u_idx, a_idx] = weight
                adj[a_idx, u_idx] = weight
                artist_weights[row['artistID']] += weight

        # 2. Process social relationships
        for _, row in user_friends.iterrows():
            u1_id = f"user_{row['userID']}"
            u2_id = f"user_{row['friendID']}"
            if u1_id in entity2id and u2_id in entity2id:
                u1_idx = entity2id[u1_id]
                u2_idx = entity2id[u2_id]
                adj[u1_idx, u2_idx] = 1
                adj[u2_idx, u1_idx] = 1

        # 3. Process user-tag relationships
        for _, row in user_tags.iterrows():
            user_id = f"user_{row['userID']}"
            tag_id = f"tag_{row['tagID']}"
            if user_id in entity2id and tag_id in entity2id:
                u_idx = entity2id[user_id]
                t_idx = entity2id[tag_id]
                self.tag2idx[row['tagID']] = t_idx
                adj[u_idx, t_idx] = 1
                adj[t_idx, u_idx] = 1

        # Normalize to get transition probabilities
        self.P = normalize(adj.tocsr(), norm='l1', axis=1)
        
        # Precompute popular artists for cold-start fallback
        self.popular_artists = sorted(artist_weights.items(),
                                    key=lambda x: -x[1])[:1000]

    def recommend(self, user_id, top_k=10):
        """Generate recommendations with cold-start fallback"""
        # Cold-start handling
        if user_id not in self.user2idx:
            return self._recommend_popular(top_k)
            
        # Standard random walk with restarts
        num_nodes = self.P.shape[0]
        x = np.zeros(num_nodes)
        x[self.user2idx[user_id]] = 1.0
        
        for _ in range(30):  # 30 iterations for convergence
            x = (1 - self.alpha) * x + self.alpha * self.P.dot(x)
        
        # Get top artists
        artist_scores = {artist: x[idx] for artist, idx in self.artist2idx.items()}
        return sorted(artist_scores.items(), key=lambda x: -x[1])[:top_k]

    def _recommend_popular(self, top_k):
        """Fallback recommendation for cold-start users"""
        return [(artist, 1.0) for artist, _ in self.popular_artists[:top_k]]

    def evaluate(self, test_user_artists, train_user_artists, top_k=10):
        """Evaluate with proper cold-start handling"""
        # Group test interactions by user
        test_users = test_user_artists.groupby('userID')['artistID'].apply(set)
        
        # Initialize metrics
        metrics = {
            'precision_sum': 0,
            'recall_sum': 0,
            'ndcg_sum': 0,
            'n_users': 0
        }
        
        for user_id, true_artists in test_users.items():
            # Get recommendations (handles cold-start automatically)
            recommended = self.recommend(user_id, top_k)
            recommended_artists = [a for a, _ in recommended]
            
            # Calculate hits
            hit_list = [1 if a in true_artists else 0 for a in recommended_artists]
            n_hits = sum(hit_list)
            
            # Update metrics
            metrics['precision_sum'] += n_hits / top_k
            metrics['recall_sum'] += n_hits / len(true_artists) if true_artists else 0
            
            # NDCG calculation
            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hit_list))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_artists), top_k)))
            metrics['ndcg_sum'] += dcg / idcg if idcg > 0 else 0
            metrics['n_users'] += 1
        
        # Finalize metrics
        results = {
            'Precision@K': metrics['precision_sum'] / metrics['n_users'],
            'Recall@K': metrics['recall_sum'] / metrics['n_users'],
            'NDCG@K': metrics['ndcg_sum'] / metrics['n_users'],
            'Users': metrics['n_users']
        }
        
        print(f"Evaluated {results['Users']} users:")
        print(f"Precision@{top_k}: {results['Precision@K']:.4f}")
        print(f"Recall@{top_k}: {results['Recall@K']:.4f}")
        print(f"NDCG@{top_k}: {results['NDCG@K']:.4f}")
        
        return results
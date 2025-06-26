import scipy.sparse as sp
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import ParameterGrid
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import networkx as nx

class RecWalk:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.user2idx = {}
        self.artist2idx = {}
        self.tag2idx = {}
        self.P = None
        self.popular_artists = None

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
        
        for _ in range(30):
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
            print(n_hits)
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

class GenreEnhancedRecWalk:
    def __init__(self, alpha=0.3, genre_weight=0.5):
        """
        Args:
            alpha: Restart probability for random walks
            genre_weight: Influence of genre similarity (0-1)
        """
        self.alpha = alpha
        self.genre_weight = genre_weight
        self.user2idx = {}       
        self.artist2idx = {}     
        self.tag2idx = {}        
        self.genre2idx = {}      
        self.id2entity = {}      
        self.P = None            
        self.artist_genres = defaultdict(list)  
        self.popular_artists = None

    def build_graph(self, user_artists, user_friends, user_tags, artist_genres, entity2id):
        """Construct the enhanced graph with genres"""
        # Process artist-genre relationships first
        self._process_genres(artist_genres, entity2id)
        
        # Initialize adjacency matrix
        num_entities = len(entity2id)
        adj = sp.lil_matrix((num_entities, num_entities))
        
        # 1. User-artist listening relationships
        artist_weights = defaultdict(float)
        for _, row in user_artists.iterrows():
            u_id = f"user_{row['userID']}"
            a_id = f"artist_{row['artistID']}"
            if u_id in entity2id and a_id in entity2id:
                u_idx = entity2id[u_id]
                a_idx = entity2id[a_id]
                
                # Store mappings
                self.user2idx[row['userID']] = u_idx
                self.artist2idx[row['artistID']] = a_idx
                
                # Weight edges by play count
                weight = np.log1p(row['weight'])
                adj[u_idx, a_idx] = weight
                adj[a_idx, u_idx] = weight
                artist_weights[row['artistID']] += weight
                
                # Connect artist to genres
                for genre in self.artist_genres.get(row['artistID'], []):
                    if genre in entity2id:
                        g_idx = entity2id[genre]
                        adj[a_idx, g_idx] = 1  
                        adj[g_idx, a_idx] = 0.5  

        # 2. Social relationships (bidirectional)
        for _, row in user_friends.iterrows():
            u1 = f"user_{row['userID']}"
            u2 = f"user_{row['friendID']}"
            if u1 in entity2id and u2 in entity2id:
                u1_idx = entity2id[u1]
                u2_idx = entity2id[u2]
                adj[u1_idx, u2_idx] = 1
                adj[u2_idx, u1_idx] = 1

        # 3. User-tag relationships
        for _, row in user_tags.iterrows():
            u_id = f"user_{row['userID']}"
            t_id = f"tag_{row['tagID']}"
            if u_id in entity2id and t_id in entity2id:
                u_idx = entity2id[u_id]
                t_idx = entity2id[t_id]
                self.tag2idx[row['tagID']] = t_idx
                adj[u_idx, t_idx] = 1
                adj[t_idx, u_idx] = 1

        # Normalize transition probabilities
        self.P = normalize(adj.tocsr(), norm='l1', axis=1)
        self.popular_artists = sorted(artist_weights.items(), 
                                    key=lambda x: -x[1])[:1000]
        
        # Create reverse ID mapping
        self.id2entity = {v: k for k, v in entity2id.items()}

    def _process_genres(self, artist_genres, entity2id):
        """Extract and index genre information"""
        for _, row in artist_genres.iterrows():
            if pd.notna(row['genres']):
                artist_id = row['artistID']
                genres = [g.strip().lower() for g in str(row['genres']).split(';') if g.strip()]
                
                for genre in genres:
                    genre_id = f"genre_{genre}"
                    self.artist_genres[artist_id].append(genre_id)
                    
                    # Add to entity2id if new
                    if genre_id not in entity2id:
                        new_id = len(entity2id)
                        entity2id[genre_id] = new_id
                        self.genre2idx[genre] = new_id

    def recommend(self, user_id, top_k=10):
        """Generate recommendations with genre awareness"""
        if user_id not in self.user2idx:
            return self._recommend_with_genres(top_k)
            
        # Perform random walk with restarts
        x = np.zeros(self.P.shape[0])
        x[self.user2idx[user_id]] = 1.0
        
        for _ in range(30):
            x = (1 - self.alpha) * x + self.alpha * self.P.dot(x)
        
        # Score artists with genre boost
        artist_scores = {}
        user_genres = self._get_user_genres(user_id)
        
        for artist, idx in self.artist2idx.items():
            base_score = x[idx]
            genre_score = self._genre_match_score(artist, user_genres)
            artist_scores[artist] = base_score * (1 + self.genre_weight * genre_score)
            
        return sorted(artist_scores.items(), key=lambda x: -x[1])[:top_k]

    def _get_user_genres(self, user_id):
        """Extract user's preferred genres from their history"""
        if user_id not in self.user2idx:
            return set()
            
        user_idx = self.user2idx[user_id]
        genre_counts = Counter()
        
        # Get all connected artists
        for artist, a_idx in self.artist2idx.items():
            if self.P[user_idx, a_idx] > 0:
                genre_counts.update(self.artist_genres.get(artist, []))
                
        return {g for g, cnt in genre_counts.most_common(5)}

    def _genre_match_score(self, artist_id, user_genres):
        """Calculate genre similarity between artist and user"""
        if not user_genres or artist_id not in self.artist_genres:
            return 0
            
        artist_genres = set(self.artist_genres[artist_id])
        return len(artist_genres & user_genres) / len(user_genres)

    def _recommend_with_genres(self, top_k):
        """Cold-start recommendation using genre popularity"""
        # Get most popular genres across all artists
        genre_counts = Counter()
        for artist, _ in self.popular_artists:
            genre_counts.update(self.artist_genres.get(artist, []))
        
        top_genres = {g for g, _ in genre_counts.most_common(3)}
        
        # Recommend artists in these genres
        recommendations = []
        for artist, weight in self.popular_artists:
            if top_genres & set(self.artist_genres.get(artist, [])):
                recommendations.append((artist, weight))
        
        return sorted(recommendations, key=lambda x: -x[1])[:top_k]

    def evaluate(self, test_data, train_data, top_k=10):
        """Enhanced evaluation with genre awareness"""
        test_users = test_data.groupby('userID')['artistID'].apply(set)
        train_artists = set(train_data['artistID'])
        
        metrics = {
            'precision': 0,
            'recall': 0,
            'ndcg': 0,
            'n_users': 0,
            'genre_coverage': 0
        }
        
        for user_id, true_artists in test_users.items():
            # Get recommendations
            recs = self.recommend(user_id, top_k)
            rec_artists = [a for a, _ in recs]
            
            # Only evaluate on artists present in training
            valid_test = true_artists & train_artists
            if not valid_test:
                continue
                
            # Precision and Recall
            hits = set(rec_artists) & valid_test
            metrics['precision'] += len(hits) / top_k
            metrics['recall'] += len(hits) / len(valid_test)
            
            # NDCG Calculation
            relevance = [1 if a in valid_test else 0 for a in rec_artists]
            
            # Calculate DCG
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            
            # Calculate IDCG
            ideal_relevance = [1]*min(len(valid_test), top_k) + [0]*max(0, top_k - len(valid_test))
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            
            metrics['ndcg'] += dcg / idcg if idcg > 0 else 0
            
            # Genre coverage
            rec_genres = set()
            for artist in rec_artists:
                rec_genres.update(self.artist_genres.get(artist, []))
            metrics['genre_coverage'] += len(rec_genres) / top_k
            
            metrics['n_users'] += 1

        # Finalize metrics
        if metrics['n_users'] == 0:
            return {
                'Precision@K': 0,
                'Recall@K': 0,
                'NDCG@K': 0,
                'GenreCoverage@K': 0,
                'Users': 0
            }
        
        results = {
            'Precision@K': metrics['precision'] / metrics['n_users'],
            'Recall@K': metrics['recall'] / metrics['n_users'],
            'NDCG@K': metrics['ndcg'] / metrics['n_users'],
            'GenreCoverage@K': metrics['genre_coverage'] / metrics['n_users'],
            'Users': metrics['n_users']
        }
        
        return results

    def save_model(self, path):
        """Save model state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'genre_weight': self.genre_weight,
                'user2idx': self.user2idx,
                'artist2idx': self.artist2idx,
                'genre2idx': self.genre2idx,
                'artist_genres': self.artist_genres,
                'P': self.P,
                'popular_artists': self.popular_artists
            }, f)

    def get_graph(self):
        """Return the networkx graph representation"""
        G = nx.DiGraph()
        
        for idx, entity in self.id2entity.items():
            if isinstance(entity, str):
                if entity.startswith('user_'):
                    G.add_node(idx, type='user', label=f"User {entity[5:]}")
                elif entity.startswith('artist_'):
                    G.add_node(idx, type='artist', label=f"Artist {entity[7:]}")
                elif entity.startswith('tag_'):
                    G.add_node(idx, type='tag', label=f"Tag {entity[4:]}")
                elif entity.startswith('genre_'):
                    G.add_node(idx, type='genre', label=f"Genre {entity[6:]}")
            else:
                continue
        
        # Add edges with weights
        rows, cols = self.P.nonzero()
        for i, j in zip(rows, cols):
            weight = self.P[i,j]
            if weight > 0.01:
                G.add_edge(i, j, weight=weight)
        
        return G

    def get_recommendation_paths(self, user_id, artist_id, max_paths=3):
        """Find important paths between user and artist"""
        if user_id not in self.user2idx or artist_id not in self.artist2idx:
            return []
        
        user_idx = self.user2idx[user_id]
        artist_idx = self.artist2idx[artist_id]
        
        # Get the top paths using the transition matrix
        paths = []
        # 1. Direct path
        if self.P[user_idx, artist_idx] > 0:
            paths.append({
                'path': [user_idx, artist_idx],
                'weight': self.P[user_idx, artist_idx],
                'type': 'direct'
            })
        
        # 2. Via friends
        for friend_id, friend_idx in self.user2idx.items():
            if (friend_id != user_id and 
                self.P[user_idx, friend_idx] > 0 and 
                self.P[friend_idx, artist_idx] > 0):
                paths.append({
                    'path': [user_idx, friend_idx, artist_idx],
                    'weight': self.P[user_idx, friend_idx] * self.P[friend_idx, artist_idx],
                    'type': 'friend'
                })
        
        # 3. Via genres/tags
        if hasattr(self, 'artist_genres'):
            for genre in self.artist_genres.get(artist_id, []):
                if genre in self.id2entity.values():
                    genre_idx = [k for k,v in self.id2entity.items() if v == genre][0]
                    if self.P[user_idx, genre_idx] > 0 and self.P[genre_idx, artist_idx] > 0:
                        paths.append({
                            'path': [user_idx, genre_idx, artist_idx],
                            'weight': self.P[user_idx, genre_idx] * self.P[genre_idx, artist_idx],
                            'type': 'genre'
                        })
        
        # Sort by weight and return top paths
        paths.sort(key=lambda x: -x['weight'])
        return paths[:max_paths]

    @classmethod
    def load_model(cls, path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['alpha'], data['genre_weight'])
        model.__dict__.update(data)
        return model

class SimilarityEnhancedRecWalk(GenreEnhancedRecWalk):
    def __init__(self, alpha=0.3, genre_weight=0.5, sim_weight=0.7):
        super().__init__(alpha, genre_weight)
        self.sim_weight = sim_weight
        self.artist_sim = None
        self.common_artists = None

    def build_graph(self, user_artists, user_friends, user_tags, artist_genres, entity2id):
        # First build the base graph
        super().build_graph(user_artists, user_friends, user_tags, artist_genres, entity2id)
        
        # Compute artist similarities
        self._compute_artist_similarities(user_artists, artist_genres)
        
        # Add similarity edges to the graph
        self._add_similarity_edges(entity2id)

    def _compute_artist_similarities(self, user_artists, artist_genres):
        """Calculate artist similarities using both listening patterns and genres"""
        # 1. Co-listening similarity - using all artists in user-artist matrix
        user_artist_matrix = user_artists.pivot_table(
            index='userID', columns='artistID', values='weight', fill_value=0
        )
        co_listen_sim = cosine_similarity(user_artist_matrix.T)
        co_listen_artists = user_artist_matrix.columns.tolist()
        
        # 2. Genre similarity - only for artists with genre information
        genre_vecs = {}
        for artist, genres in artist_genres.items():
            if artist in self.artist2idx:
                genre_vecs[artist] = Counter(genres)
        
        genre_artists = sorted(genre_vecs.keys())
        genre_sim = np.zeros((len(genre_artists), len(genre_artists)))
        for i, a1 in enumerate(genre_artists):
            for j, a2 in enumerate(genre_artists):
                if i <= j:
                    genre_sim[i,j] = self._jaccard_similarity(genre_vecs[a1], genre_vecs[a2])
                    genre_sim[j,i] = genre_sim[i,j]
        
        # Find intersection of artists that appear in both
        common_artists = sorted(set(co_listen_artists) & set(genre_artists) & set(self.artist2idx.keys()))
        
        # Create mapping from artist to index in each matrix
        co_listen_idx = {a: i for i, a in enumerate(co_listen_artists)}
        genre_idx = {a: i for i, a in enumerate(genre_artists)}
        common_idx = {a: i for i, a in enumerate(common_artists)}
        
        # Create combined similarity matrix only for common artists
        n = len(common_artists)
        combined_sim = np.zeros((n, n))
        
        for i, a1 in enumerate(common_artists):
            for j, a2 in enumerate(common_artists):
                if i <= j:
                    # Get indices in each matrix
                    cl_i, cl_j = co_listen_idx[a1], co_listen_idx[a2]
                    g_i, g_j = genre_idx[a1], genre_idx[a2]
                    
                    # Combine similarities (weighted average)
                    combined_sim[i,j] = 0.6 * co_listen_sim[cl_i, cl_j] + 0.4 * genre_sim[g_i, g_j]
                    combined_sim[j,i] = combined_sim[i,j]
        
        # Store the similarity matrix and mapping
        self.artist_sim = combined_sim
        self.common_artists = common_artists
        self.common_idx = common_idx
        np.fill_diagonal(self.artist_sim, 0)

    def _jaccard_similarity(self, a, b):
        """Calculate Jaccard similarity between two genre sets"""
        intersection = sum((a & b).values())
        union = sum((a | b).values())
        return intersection / union if union else 0

    def _add_similarity_edges(self, entity2id):
        """Add similarity edges to the transition matrix"""
        if not hasattr(self, 'common_artists') or not self.common_artists:
            return
            
        artist_indices = {a: entity2id[f"artist_{a}"] for a in self.common_artists}
        
        # For each artist, connect to top-k most similar artists
        for artist in self.common_artists:
            if artist not in self.common_idx:
                continue
                
            i = self.common_idx[artist]
            sim_scores = self.artist_sim[i]
            
            # Get top similar artists (excluding self)
            top_k = min(5, len(sim_scores)-1)
            top_indices = np.argpartition(sim_scores, -top_k)[-top_k:]
            
            for similar_idx in top_indices:
                similar_artist = self.common_artists[similar_idx]
                sim_score = sim_scores[similar_idx]
                
                if sim_score > 0.3 and similar_artist in artist_indices:
                    a_idx = artist_indices[artist]
                    s_idx = artist_indices[similar_artist]
                    
                    # Add bidirectional edges with similarity weight
                    self.P[a_idx, s_idx] += self.sim_weight * sim_score
                    self.P[s_idx, a_idx] += self.sim_weight * sim_score
        
        # Renormalize the transition matrix
        self.P = normalize(self.P, norm='l1', axis=1)

    def _rerank_by_similarity(self, recommendations, user_artists):
        """Re-rank recommendations based on similarity to user's artists"""
        if not user_artists or not hasattr(self, 'common_idx'):
            return recommendations
            
        scored_recs = []
        for artist, score in recommendations:
            if artist not in self.common_idx:
                scored_recs.append((artist, score))
                continue
                
            sim_score = 0
            artist_idx = self.common_idx[artist]
            
            for user_artist in user_artists:
                if user_artist in self.common_idx:
                    user_artist_idx = self.common_idx[user_artist]
                    sim_score += self.artist_sim[artist_idx, user_artist_idx]
            
            avg_sim = sim_score / len(user_artists) if user_artists else 0
            # Boost original score by similarity (50% of similarity score)
            scored_recs.append((artist, score * (1 + 0.5 * avg_sim)))  
            
        return sorted(scored_recs, key=lambda x: -x[1])

    
    def _explain_recommendations(self, user_id, recommendations):
        """Generate explanations for recommendations"""
        explanations = []
        user_idx = self.user2idx[user_id]
        
        for artist, score in recommendations:
            explanation = {
                'artist': artist,
                'score': score,
                'sources': []
            }
            
            # 1. Check direct user-artist connections
            artist_idx = self.artist2idx[artist]
            if self.P[user_idx, artist_idx] > 0:
                explanation['sources'].append({
                    'type': 'direct_listening',
                    'weight': self.P[user_idx, artist_idx]
                })
            
            # 2. Check friend connections
            for friend in self.user2idx:
                if friend != user_id and self.P[self.user2idx[friend], artist_idx] > 0:
                    explanation['sources'].append({
                        'type': 'friend_listening',
                        'friend': friend,
                        'weight': self.P[self.user2idx[friend], artist_idx]
                    })
            
            # 3. Check genre connections
            if hasattr(self, 'artist_genres') and artist in self.artist_genres:
                for genre in self.artist_genres[artist]:
                    explanation['sources'].append({
                        'type': 'genre',
                        'genre': genre,
                        'weight': self.genre_weight
                    })
            
            # 4. Check similarity connections
            if artist in self.common_idx:
                artist_sim_idx = self.common_idx[artist]
                similar_artists = [
                    (self.common_artists[i], sim) 
                    for i, sim in enumerate(self.artist_sim[artist_sim_idx])
                    if sim > 0.3 and i != artist_sim_idx
                ]
                for sim_artist, sim_score in similar_artists[:3]:
                    explanation['sources'].append({
                        'type': 'similar_artist',
                        'artist': sim_artist,
                        'similarity': sim_score,
                        'weight': self.sim_weight * sim_score
                    })
            
            explanations.append(explanation)
        
        return explanations

    def recommend(self, user_id, top_k=10, explain=False):
        """Enhanced recommendation with similarity propagation and explanation support"""
        recs = super().recommend(user_id, top_k*2)
        
        # Re-rank using similarity to user's preferred artists
        if user_id in self.user2idx:
            user_artists = [a for a in self.artist2idx 
                          if self.P[self.user2idx[user_id], self.artist2idx[a]] > 0
                          and a in self.common_idx]
            if user_artists:
                recs = self._rerank_by_similarity(recs, user_artists)
        
        final_recs = recs[:top_k]
        
        if explain:
            return self._explain_recommendations(user_id, final_recs)
        return final_recs

    def get_kg_paths(self, user_id, artist_id, max_paths=3, min_weight=0.0):
        """Get important KG paths that connect user to recommended artist with consistent keys"""
        paths = []
        
        # Check if user and artist exist in the graph first
        if user_id not in self.user2idx:
            print(f"Debug: User {user_id} not found in graph")
            return []
            
        if artist_id not in self.artist2idx:
            print(f"Debug: Artist {artist_id} not found in graph")
            return []
        
        user_idx = self.user2idx[user_id]
        artist_idx = self.artist2idx[artist_id]
        
        # 1. Direct connection path (user -> artist)
        direct_weight = self.P[user_idx, artist_idx]
        if direct_weight >= min_weight:
            paths.append({
                'path_type': 'direct',
                'path': ['user', 'listened', 'artist'],
                'weight': direct_weight,
                'details': {
                    'direct_weight': direct_weight
                }
            })
        
        # 2. Friend path (user -> friend -> artist)
        for friend_id, friend_idx in self.user2idx.items():
            if friend_id == user_id:
                continue
                
            user_friend = self.P[user_idx, friend_idx]
            friend_artist = self.P[friend_idx, artist_idx]
            if user_friend > 0 and friend_artist > 0:
                combined_weight = user_friend * friend_artist
                if combined_weight >= min_weight:
                    paths.append({
                        'path_type': 'friend',
                        'path': ['user', 'friends_with', 'user', 'listened', 'artist'],
                        'weight': combined_weight,
                        'details': {
                            'friend_id': friend_id,
                            'user_friend_weight': user_friend,
                            'friend_artist_weight': friend_artist
                        }
                    })
        
        # 3. Genre path (user -> genre -> artist)
        if hasattr(self, 'artist_genres') and artist_id in self.artist_genres:
            # Get user's top genres from their artists
            user_genres = Counter()
            for art_id, art_idx in self.artist2idx.items():
                if self.P[user_idx, art_idx] > 0 and art_id in self.artist_genres:
                    user_genres.update(self.artist_genres[art_id])
            
            # Check each genre of target artist
            for genre in self.artist_genres[artist_id]:
                if genre in user_genres:
                    genre_weight = self.genre_weight * (1 + user_genres[genre])
                    paths.append({
                        'path_type': 'genre',
                        'path': ['user', 'prefers_genre', 'genre', 'describes', 'artist'],
                        'weight': genre_weight,
                        'details': {
                            'genre': genre,
                            'user_genre_count': user_genres[genre]
                        }
                    })
        
        # 4. Similarity path (user -> similar artist -> artist)
        if hasattr(self, 'common_idx') and artist_id in self.common_idx:
            artist_sim_idx = self.common_idx[artist_id]
            
            # Find artists user listens to that are similar to target
            user_artists = [
                a for a in self.artist2idx 
                if self.P[user_idx, self.artist2idx[a]] > 0 
                and a in self.common_idx
            ]
            
            for u_artist in user_artists:
                if u_artist == artist_id:
                    continue
                    
                u_idx = self.common_idx[u_artist]
                sim_score = self.artist_sim[u_idx, artist_sim_idx]
                if sim_score > 0:
                    combined_weight = (self.P[user_idx, self.artist2idx[u_artist]] * 
                                    self.sim_weight * 
                                    sim_score)
                    if combined_weight >= min_weight:
                        paths.append({
                            'path_type': 'similarity',
                            'path': ['user', 'listened', 'artist', 'similar_to', 'artist'],
                            'weight': combined_weight,
                            'details': {
                                'via_artist': u_artist,
                                'similarity_score': sim_score,
                                'user_artist_weight': self.P[user_idx, self.artist2idx[u_artist]]
                            }
                        })
        
        # Sort by weight and return top paths
        paths.sort(key=lambda x: -x['weight'])
        return paths[:max_paths]

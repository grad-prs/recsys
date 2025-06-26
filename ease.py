from scipy.sparse import csr_matrix, diags, issparse
from scipy import sparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class EASE:
    def __init__(self, reg_weight=100.0):
        self.reg_weight = reg_weight
        self.item_similarity = None
        self.item_ids = None
    
    def fit(self, user_artist_df):
        """Train EASE model with weighted interactions"""
        users = user_artist_df['userID'].unique()
        artists = user_artist_df['artistID'].unique()
        self.item_ids = {aid: i for i, aid in enumerate(artists)}
        
        # Construct weighted interaction matrix
        row = user_artist_df['userID'].map({uid: i for i, uid in enumerate(users)}).values
        col = user_artist_df['artistID'].map(self.item_ids).values
        data = user_artist_df['weight'].values
        X = csr_matrix((data, (row, col)), shape=(len(users), len(artists)))
        
        # Original paper's efficient formulation (Eq. 3)
        G = X.T @ X
        diag = diags([self.reg_weight] * G.shape[0], format='csr')
        B = np.linalg.inv(G + diag.toarray() + 1e-10 * np.eye(G.shape[0]))
        
        # Compute similarity matrix (Eq. 2)
        self.item_similarity = -B / np.diag(B)[:, None]
        np.fill_diagonal(self.item_similarity, 0.0)
    
    def recommend(self, user_id, user_artist_df, top_k=10):
        """Generate weighted recommendations"""
        if self.item_similarity is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        user_data = user_artist_df[user_artist_df['userID'] == user_id]
        if len(user_data) == 0:
            return []  # Cold-start user
            
        # Get user's artists and weights
        user_artists = []
        user_weights = []
        for _, row in user_data.iterrows():
            if row['artistID'] in self.item_ids:
                user_artists.append(self.item_ids[row['artistID']])
                user_weights.append(row['weight'])
        
        if not user_artists:
            return []
            
        # Compute scores (weighted sum of similarities)
        scores = np.zeros((1, len(self.item_ids)))
        for art_idx, weight in zip(user_artists, user_weights):
            scores += self.item_similarity[art_idx] * weight
        scores = scores.squeeze()
        # Get top-K recommendations
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [
            (list(self.item_ids.keys())[i], float(scores[i]))
            for i in top_indices
        ]
    
    def evaluate(self, test_user_artists, train_user_artists, top_k=10):
        """Evaluation with weighted interactions"""
        test_users = test_user_artists.groupby('userID')['artistID'].apply(set)
        train_artists = set(train_user_artists['artistID'])
        
        metrics = {
            'precision': 0,
            'recall': 0,
            'ndcg': 0,
            'n_users': 0
        }
        
        for user_id, true_artists in test_users.items():
            recommended = self.recommend(user_id, train_user_artists, top_k)
            if not recommended:
                continue
                
            rec_artists = {a for a, _ in recommended}
            valid_test = true_artists & train_artists
            
            if not valid_test:
                continue
                
            # Precision and Recall
            hits = rec_artists & valid_test
            metrics['precision'] += len(hits) / top_k
            metrics['recall'] += len(hits) / len(valid_test)
            
            # NDCG (using weights if available)
            relevance = {a: w for a, w in recommended if a in valid_test}
            dcg = sum(w / np.log2(i + 2) for i, (a, w) in enumerate(recommended) if a in relevance)
            idcg = sum(sorted(relevance.values(), reverse=True)[i] / np.log2(i + 2) 
                   for i in range(min(len(relevance), top_k)))
            metrics['ndcg'] += dcg / idcg if idcg > 0 else 0
            metrics['n_users'] += 1
        
        if metrics['n_users'] == 0:
            return {'Precision@K': 0, 'Recall@K': 0, 'NDCG@K': 0, 'Users': 0}
            
        return {
            'Precision@K': metrics['precision'] / metrics['n_users'],
            'Recall@K': metrics['recall'] / metrics['n_users'],
            'NDCG@K': metrics['ndcg'] / metrics['n_users'],
            'Users': metrics['n_users']
        }
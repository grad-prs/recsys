import torch
import pickle
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

class HybridRecommender:
    def __init__(self, model_dir, default_r_fa=0.5, default_r_ta=0.5, k=10):
        self.model_dir = model_dir
        self.k = k
        self.default_r_fa = default_r_fa
        self.default_r_ta = default_r_ta
        self._load_data()

    def _load_data(self):
        # Load embeddings and mappings
        self.user_embs = torch.load(f"{self.model_dir}/user_embs.pt")
        self.artist_embs = torch.load(f"{self.model_dir}/artist_embs.pt")
        self.tag_embs = torch.load(f"{self.model_dir}/tag_embs.pt")

        with open(f"{self.model_dir}/user2idx.pkl", "rb") as f:
            self.user2idx = pickle.load(f)
        with open(f"{self.model_dir}/artist2idx.pkl", "rb") as f:
            self.artist2idx = pickle.load(f)
        with open(f"{self.model_dir}/tag2idx.pkl", "rb") as f:
            self.tag2idx = pickle.load(f)
        with open(f"{self.model_dir}/user_ratios.pkl", "rb") as f:
            self.user_ratios = pickle.load(f)

        self.artist_ids = list(self.artist2idx.keys())

    def _compute_weights(self, friend_info, tag_info, r_fa, r_ta):
        if not friend_info:
            r_fa = 0.0
        if not tag_info:
            r_ta = 0.0
        denom = 1.0 + r_fa + r_ta
        alpha = 1.0 / denom
        beta = r_fa * alpha
        gamma = r_ta * alpha
        return alpha, beta, gamma

    def _get_scores(self, user_name, user_friends, user_tags, r_fa, r_ta):
        if user_name not in self.user2idx:
            return None

        user_idx = self.user2idx[user_name]
        user_id = int(user_name.split("_")[1])

        # Friend and tag availability
        friends = user_friends.get(user_id, [])
        tags = user_tags.get(user_id, [])

        friend_info = len(friends) > 0
        tag_info = len(tags) > 0

        alpha, beta, gamma = self._compute_weights(friend_info, tag_info, r_fa, r_ta)

        personal_score = torch.matmul(self.artist_embs, self.user_embs[user_idx])

        friend_score = torch.zeros_like(personal_score)
        if friend_info:
            friend_idxs = [self.user2idx[f"user_{fid}"] for fid in friends if f"user_{fid}" in self.user2idx]
            if friend_idxs:
                friend_embs = self.user_embs[friend_idxs]
                friend_score = torch.matmul(self.artist_embs, friend_embs.mean(dim=0))

        tag_score = torch.zeros_like(personal_score)
        if tag_info:
            tag_idxs = [self.tag2idx[f"tag_{tid}"] for tid in tags if f"tag_{tid}" in self.tag2idx]
            if tag_idxs:
                tag_embs = self.tag_embs[tag_idxs]
                tag_score = torch.matmul(self.artist_embs, tag_embs.mean(dim=0))

        return alpha * personal_score + beta * friend_score + gamma * tag_score

    def recommend(self, user_name, user_friends, user_tags, personalized=False):
        if personalized and user_name in self.user_ratios:
            r_fa = self.user_ratios[user_name]["r_fa"]
            r_ta = self.user_ratios[user_name]["r_ta"]
        else:
            r_fa, r_ta = self.default_r_fa, self.default_r_ta

        scores = self._get_scores(user_name, user_friends, user_tags, r_fa, r_ta)
        if scores is None:
            return []

        topk = torch.topk(scores, k=self.k)
        return [self.artist_ids[i] for i in topk.indices]

    def evaluate(self, user_artists, user_friends, user_tags, personalized=False):
        ground_truth = defaultdict(list)
        for user_id, artist_id in user_artists:
            user_key = f"user_{user_id}"
            artist_key = f"artist_{artist_id}"
            if user_key in self.user2idx and artist_key in self.artist2idx:
                ground_truth[self.user2idx[user_key]].append(self.artist2idx[artist_key])

        popularity = Counter([a for arts in ground_truth.values() for a in arts])

        precision_list, recall_list, ndcg_list, nscg_list = [], [], [], []

        for user_idx, true_artists in ground_truth.items():
            user_name = list(self.user2idx.keys())[list(self.user2idx.values()).index(user_idx)]
            recs = self.recommend(user_name, user_friends, user_tags, personalized)
            rec_indices = [self.artist2idx[a] for a in recs if a in self.artist2idx]

            if not rec_indices or not true_artists:
                continue

            # Metrics
            hits = sum(1 for a in rec_indices if a in true_artists)
            precision = hits / self.k
            recall = hits / len(true_artists)

            relevance = [1 if a in true_artists else 0 for a in rec_indices]
            dcg = sum([r / np.log2(i + 2) for i, r in enumerate(relevance)])
            idcg = sum([r / np.log2(i + 2) for i, r in enumerate(sorted(relevance, reverse=True))])
            ndcg = dcg / idcg if idcg > 0 else 0.0

            novelty = [1 / np.log2(popularity.get(a, 1) + 2) for a in rec_indices]
            score = sum(r * n for r, n in zip(relevance, novelty))
            max_score = sum(sorted(relevance, reverse=True)[i] * sorted(novelty, reverse=True)[i]
                            for i in range(len(relevance)))
            nscg = score / max_score if max_score > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            nscg_list.append(nscg)

        return {
            "Precision@K": np.mean(precision_list),
            "Recall@K": np.mean(recall_list),
            "NDCG@K": np.mean(ndcg_list),
            "NSCG@K": np.mean(nscg_list)
        }

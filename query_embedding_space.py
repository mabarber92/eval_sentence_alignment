from sentence_transformers import SentenceTransformer
import numpy as np
from umap import UMAP
import pandas as pd
import plotly.express as px
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

class embeddingSpace():

    _SUPPORTED_REDUCERS = ("umap", "pacmap")

    def __init__(self, eval_pairs, embedding_models, cache_path=None):
        """Initialise using an eval_pairs object, load the models and create the embedding space.
        If cache_path is provided, loads cached embeddings and only runs models not already cached."""
        self.eval_pairs = eval_pairs
        self.cache_path = cache_path
        self._d_reduction = "umap"
        self.run_embedding_models(embedding_models)

    def set_d_reduction(self, method):
        """Set the dimensionality reduction algorithm for visualisation. Options: 'umap', 'pacmap'."""
        if method not in self._SUPPORTED_REDUCERS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self._SUPPORTED_REDUCERS}")
        self._d_reduction = method

    def initialise_model(self, embed_model, model_kwargs=None):
        return SentenceTransformer(embed_model, device="cuda", model_kwargs=model_kwargs or {})

    def _build_sentence_index(self):
        """Build a df mapping each flat embedding index to group_id, list_position, sentence, is_valid.
        Row i of any embedding array corresponds to sentence_index.iloc[i]."""
        rows = []
        for group_id, group in enumerate(self.eval_pairs.eval_groups):
            for list_position, sentence in enumerate(group):
                rows.append({
                    "group_id": group_id,
                    "list_position": list_position,
                    "sentence": sentence,
                    "is_valid": pd.notna(sentence)
                })
        self.sentence_index = pd.DataFrame(rows)

    def encode_groups(self, model, prefix=""):
        valid_mask = self.sentence_index["is_valid"].values
        valid_sentences = self.sentence_index.loc[valid_mask, "sentence"].tolist()

        if prefix:
            valid_sentences = [prefix + s for s in valid_sentences]

        valid_embeddings = model.encode(valid_sentences, show_progress_bar=True)

        result = np.zeros((len(self.sentence_index), valid_embeddings.shape[1]))
        result[valid_mask] = valid_embeddings

        return result

    @staticmethod
    def _parse_model_entry(entry):
        """Accept either a plain model ID string or a dict with 'model', optional 'prefix', and optional 'model_kwargs'."""
        if isinstance(entry, dict):
            return entry["model"], entry.get("prefix", ""), entry.get("model_kwargs", {})
        return entry, "", {}

    def run_embedding_models(self, embedding_models):
        """Encodes the groups with a series of embedding models so
        that the performance of the models can be compared.
        If a cache_path is set, loads existing embeddings and skips models already cached."""
        self._build_sentence_index()

        if self.cache_path:
            try:
                data = np.load(self.cache_path)
                self.embedding_spaces = {key: data[key] for key in data.files}
            except FileNotFoundError:
                self.embedding_spaces = {}
        else:
            self.embedding_spaces = {}

        for entry in embedding_models:
            model_id, prefix, model_kwargs = self._parse_model_entry(entry)
            model_name = model_id.split("/")[-1]

            if model_name in self.embedding_spaces:
                print(f"Loaded cached embeddings for {model_name}")
                continue

            print(f"Creating embeddings with {model_name}")
            model = self.initialise_model(model_id, model_kwargs=model_kwargs)
            self.embedding_spaces[model_name] = self.encode_groups(model, prefix=prefix)

        if self.cache_path:
            self.save()

    def save(self):
        """Save all embedding spaces to the cache_path as a .npz file."""
        np.savez_compressed(self.cache_path, **self.embedding_spaces)

    # Functions for running the evaluations on the given embedding set

    def add_2d_dimensions(self, df):
        """Take a flattened df and add x y coords for a 2d representation of the embeds.
        Invalid (zero vector) rows are excluded from reduction and get NaN coordinates.
        Uses the algorithm set via set_d_reduction (default: umap)."""
        valid_mask = df["is_valid"]
        valid_embeddings = np.stack(df.loc[valid_mask, "embed"].tolist())

        if self._d_reduction == "umap":
            reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42)
            coords = reducer.fit_transform(valid_embeddings)
        elif self._d_reduction == "pacmap":
            import pacmap
            reducer = pacmap.PaCMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(valid_embeddings)

        df["x"] = np.nan
        df["y"] = np.nan
        df.loc[valid_mask, "x"] = coords[:, 0]
        df.loc[valid_mask, "y"] = coords[:, 1]

        return df

    def flatten_to_df(self, embeddings, labels=None):
        """Return a df where each row is a sentence with its embedding and group membership.
        labels is an optional list mapping list_position to a human-readable name."""
        df = self.sentence_index.copy()
        df["embed"] = list(embeddings)
        if labels is not None:
            df["label"] = df["list_position"].apply(lambda p: labels[p] if p < len(labels) else None)
        return df

    def _select_models(self, top_models=None, bottom_models=None):
        """Return an ordered list of model names filtered to top/bottom N by mean recall.
        If both are None, returns all models."""
        if top_models is None and bottom_models is None:
            return list(self.embedding_spaces.keys())

        recall_col = f"recall_at_{self._top_n}"
        model_recall = (
            self.eval_summary.groupby("model")[recall_col].mean()
            .sort_values(ascending=False)
        )
        selected = []
        if top_models:
            selected += model_recall.head(top_models).index.tolist()
        if bottom_models:
            for m in model_recall.tail(bottom_models).index.tolist():
                if m not in selected:
                    selected.append(m)
        return selected

    def _build_plot_df(self, labels, group_ids=None, models=None):
        """Build the UMAP-reduced df, optionally filtered to group_ids and a subset of models."""
        full_df = pd.DataFrame()
        model_items = {k: v for k, v in self.embedding_spaces.items() if models is None or k in models}
        for model_name, embedding_space in model_items.items():
            df = self.flatten_to_df(embedding_space, labels)
            df["model"] = model_name
            df = self.add_2d_dimensions(df)
            full_df = pd.concat([full_df, df], ignore_index=True)

        full_df = full_df[full_df["is_valid"]]
        if group_ids is not None:
            full_df = full_df[full_df["group_id"].isin(group_ids)]
        full_df["group_id"] = full_df["group_id"].astype(str)
        return full_df

    def graph_embedding_space(self, labels, html_out, group_ids=None):
        """Produce a graph of the embedding space faceted on the embedding model.
        Colour encodes group membership; symbol encodes list position (via labels).
        group_ids restricts to a specific subset of groups, shared across all facets —
        use eval_scores to select passing/failing/edge-case groups before calling."""
        full_df = self._build_plot_df(labels, group_ids=group_ids)
        fig = px.scatter(
            full_df, x="x", y="y",
            color="group_id",
            symbol="label" if "label" in full_df.columns else None,
            hover_data=["sentence"],
            facet_col="model",
            height=600, width=500 * full_df["model"].nunique()
        )
        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None)
        fig.write_html(html_out)

    def graph_eval_sample(self, labels, html_out, n_pass=5, n_fail=5, n_mixed=5, top_models=None, bottom_models=None):
        """Sample passing, failing, and mixed groups and draw a demo-ready graph.
        Symbol is per-model: circle = passes in that model, cross = fails.
        top_models/bottom_models: select N best/worst models by recall for faceting.
        Facet titles include recall@k score. Requires evaluate_pairwise_cosine to have been run first."""
        if not hasattr(self, "eval_scores"):
            raise RuntimeError("Run evaluate_pairwise_cosine before graphing eval results.")

        evaluable = self.eval_scores[self.eval_scores["has_true_pair"]]

        # Overall pass rate per group (across all models and list_pairs) for pass/fail sampling
        overall_pass_rate = evaluable.groupby("query_group_id")["in_top_n"].mean()
        passing = overall_pass_rate[overall_pass_rate == 1.0].index.values
        failing = overall_pass_rate[overall_pass_rate == 0.0].index.values

        # Mixed: select groups with the largest rank spread across models — genuine disagreement
        rank_spread = (
            evaluable.groupby("query_group_id")["true_pair_rank"]
            .agg(lambda x: x.max() - x.min())
            .rename("rank_spread")
        )
        mixed_candidates = rank_spread[
            overall_pass_rate[(overall_pass_rate > 0.0) & (overall_pass_rate < 1.0)].index
        ].sort_values(ascending=False)

        sampled_pass = np.random.choice(passing, size=min(n_pass, len(passing)), replace=False)
        sampled_fail = np.random.choice(failing, size=min(n_fail, len(failing)), replace=False)
        sampled_mixed = mixed_candidates.head(n_mixed).index.values
        selected = np.concatenate([sampled_pass, sampled_fail, sampled_mixed])

        selected_models = self._select_models(top_models=top_models, bottom_models=bottom_models)
        full_df = self._build_plot_df(labels, group_ids=selected, models=selected_models)

        # Per-(group_id, model) pass/fail — each facet gets its own symbol per group
        model_results = (
            evaluable.groupby(["model", "query_group_id"])["in_top_n"]
            .mean()
            .reset_index()
            .rename(columns={"query_group_id": "group_id", "in_top_n": "pass_rate"})
        )
        model_results["result"] = model_results["pass_rate"].apply(
            lambda x: "pass" if x == 1.0 else "fail"
        )
        model_results["group_id"] = model_results["group_id"].astype(str)
        full_df = full_df.merge(model_results[["model", "group_id", "result"]], on=["model", "group_id"], how="left")

        # Rename model column to include group-level recall in facet title
        # group_recall_at_k requires ALL list_pair comparisons to pass — stricter than per-pair recall
        group_recall_col = f"group_recall_at_{self._top_n}"
        model_recall = self.eval_summary.groupby("model")[group_recall_col].first()
        full_df["model"] = full_df["model"].apply(
            lambda m: f"{m}<br>group recall@{self._top_n}: {model_recall[m]:.0%}"
        )

        fig = px.scatter(
            full_df, x="x", y="y",
            color="group_id",
            symbol="result",
            symbol_map={"pass": "circle", "fail": "x"},
            hover_data=["sentence"],
            facet_col="model",
            height=600, width=500 * full_df["model"].nunique()
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None)
        fig.write_html(html_out)

    def run_full_evaluation(self, labels, output_dir, top_n=5, n_pass=5, n_fail=5, n_mixed=5, top_models=None, bottom_models=None):
        """Run all evaluation steps and save results to output_dir.
        Skips pairwise cosine and evaluation if already computed (e.g. after loading from cache)."""
        os.makedirs(output_dir, exist_ok=True)

        if not hasattr(self, "eval_scores"):
            self.evaluate_pairwise_cosine(top_n=top_n)

        self.eval_scores.to_csv(os.path.join(output_dir, "eval_scores.csv"), index=False, encoding="utf-8-sig")
        self.eval_summary.to_csv(os.path.join(output_dir, "eval_summary.csv"), index=False, encoding="utf-8-sig")
        self.graph_eval_sample(labels, os.path.join(output_dir, "embedding_space.html"),
                               n_pass=n_pass, n_fail=n_fail, n_mixed=n_mixed,
                               top_models=top_models, bottom_models=bottom_models)

    def run_pairwise_cosine(self):
        """Run exhaustive pairwise cosine similarity between all list-position pairs across all models.
        Stores results in self.pairwise_results as {model_name: df}."""
        list_positions = sorted(self.sentence_index["list_position"].unique())
        position_pairs = list(combinations(list_positions, 2))

        self.pairwise_results = {}

        for model_name, embeddings in self.embedding_spaces.items():
            chunks = []

            for pos_a, pos_b in tqdm(position_pairs):
                mask_a = (self.sentence_index["list_position"] == pos_a) & self.sentence_index["is_valid"]
                mask_b = (self.sentence_index["list_position"] == pos_b) & self.sentence_index["is_valid"]

                meta_a = self.sentence_index[mask_a].reset_index()
                meta_b = self.sentence_index[mask_b].reset_index()

                emb_a = embeddings[meta_a["index"].values]
                emb_b = embeddings[meta_b["index"].values]

                sim_matrix = cosine_similarity(emb_a, emb_b)

                i_idx, j_idx = np.meshgrid(np.arange(len(meta_a)), np.arange(len(meta_b)), indexing="ij")
                i_flat, j_flat = i_idx.ravel(), j_idx.ravel()

                chunks.append(pd.DataFrame({
                    "list_pair": f"{pos_a}v{pos_b}",
                    "query_group_id": meta_a.iloc[i_flat]["group_id"].values,
                    "query_sentence": meta_a.iloc[i_flat]["sentence"].values,
                    "candidate_group_id": meta_b.iloc[j_flat]["group_id"].values,
                    "candidate_sentence": meta_b.iloc[j_flat]["sentence"].values,
                    "cosine_similarity": sim_matrix.ravel(),
                    "is_true_pair": meta_a.iloc[i_flat]["group_id"].values == meta_b.iloc[j_flat]["group_id"].values
                }))

            self.pairwise_results[model_name] = pd.concat(chunks, ignore_index=True)

    def evaluate_pairwise_cosine(self, top_n=5, eval_csv=None):
        self._top_n = top_n
        """For each query sentence, check if the true pair appears in the top_n candidates by cosine similarity.
        Stores self.eval_scores (per-query) and self.eval_summary (per-model recall@k and MRR).
        Returns (eval_scores, eval_summary)."""

        if not hasattr(self, "pairwise_results"):
            self.run_pairwise_cosine()

        eval_rows = []
        for model_name, pairwise_df in self.pairwise_results.items():
            for (list_pair, query_group_id, query_sentence), query_df in pairwise_df.groupby(
                ["list_pair", "query_group_id", "query_sentence"]
            ):
                ranked = query_df.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
                true_pair_positions = ranked.index[ranked["is_true_pair"]].tolist()

                rank = true_pair_positions[0] + 1 if true_pair_positions else None  # 1-indexed
                has_true_pair = rank is not None  # rank is None iff counterpart was None/filtered
                eval_rows.append({
                    "model": model_name,
                    "list_pair": list_pair,
                    "query_group_id": query_group_id,
                    "query_sentence": query_sentence,
                    "has_true_pair": has_true_pair,
                    "true_pair_rank": rank,
                    "in_top_n": has_true_pair and rank <= top_n,
                    "reciprocal_rank": 1 / rank if has_true_pair else None
                })

        self.eval_scores = pd.DataFrame(eval_rows)

        # Count how many models passed for each (group, list_pair) — agreement signal
        agreement = (
            self.eval_scores.groupby(["query_group_id", "list_pair"])["in_top_n"]
            .sum()
            .reset_index()
            .rename(columns={"in_top_n": "models_passing"})
        )
        self.eval_scores = self.eval_scores.merge(agreement, on=["query_group_id", "list_pair"])

        # Summary: recall@1..top_n and MRR per model and list_pair
        # Precision@k is omitted — with 1 true pair per query it equals recall@k / k (no additional signal)
        # Recall and MRR are computed only over queries where a true pair exists (has_true_pair=True).
        # Queries where the counterpart was None are excluded from the denominator, not penalised.
        evaluable = self.eval_scores[self.eval_scores["has_true_pair"]]

        def _summarise(df):
            row = {
                "n_evaluable": len(df),
                "mrr": df["reciprocal_rank"].mean(),
                "mean_rank": df["true_pair_rank"].mean(),
            }
            for k in range(1, top_n + 1):
                row[f"recall_at_{k}"] = (df["true_pair_rank"] <= k).mean()
            return pd.Series(row)

        self.eval_summary = (
            evaluable.groupby(["model", "list_pair"])
            .apply(_summarise)
            .reset_index()
        )

        # Group-level recall: a group only passes if ALL its list_pair comparisons pass.
        # More punitive than per-pair recall — reflects whether the full eval group was resolved.
        for k in range(1, top_n + 1):
            group_recall = (
                evaluable.assign(hit=evaluable["true_pair_rank"] <= k)
                .groupby(["model", "query_group_id"])["hit"]
                .all()
                .groupby(level="model")
                .mean()
                .rename(f"group_recall_at_{k}")
                .reset_index()
            )
            self.eval_summary = self.eval_summary.merge(group_recall, on="model", how="left")

        if eval_csv is not None:
            self.eval_scores.to_csv(eval_csv, index=False, encoding="utf-8-sig")

        return self.eval_scores, self.eval_summary

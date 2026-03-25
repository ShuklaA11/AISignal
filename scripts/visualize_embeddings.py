"""
Embedding Space Visualization
Generates t-SNE and UMAP-style plots of article embeddings colored by topic/category.
Shows that the embedding space captures meaningful semantic structure.
"""

import sys
sys.path.insert(0, "/Users/arnavshukla/aisignal")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sqlmodel import Session, select, create_engine
from src.storage.models import ArticleEmbedding, Article
from collections import Counter

DB_URL = "sqlite:////Users/arnavshukla/aisignal/data/newsletter.db"
OUTPUT_DIR = "/Users/arnavshukla/aisignal/scripts"


def load_data():
    """Load all article embeddings and metadata."""
    engine = create_engine(DB_URL)
    with Session(engine) as session:
        # Load all embeddings
        emb_records = session.exec(select(ArticleEmbedding)).all()
        article_ids = [r.article_id for r in emb_records]
        embeddings = np.array([
            np.frombuffer(r.embedding_blob, dtype=np.float32) for r in emb_records
        ])

        # Load article metadata
        articles = session.exec(
            select(Article).where(Article.id.in_(article_ids))
        ).all()
        article_map = {a.id: a for a in articles}

    # Build aligned arrays
    valid_ids = []
    valid_embeddings = []
    primary_topics = []
    categories = []
    sources = []

    for i, aid in enumerate(article_ids):
        if aid not in article_map:
            continue
        a = article_map[aid]
        topics = a.topics
        valid_ids.append(aid)
        valid_embeddings.append(embeddings[i])
        primary_topics.append(topics[0] if topics else "Unknown")
        categories.append(a.category or "Unknown")
        sources.append(a.source_name or "Unknown")

    return (
        np.array(valid_embeddings),
        primary_topics,
        categories,
        sources,
    )


def run_tsne(embeddings, perplexity=30, seed=42):
    """Reduce to 2D with t-SNE."""
    print(f"Running t-SNE on {len(embeddings)} embeddings (1024-dim → 2-dim)...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def plot_by_label(coords, labels, title, filename, figsize=(14, 10), max_legend=15):
    """Scatter plot colored by label."""
    label_counts = Counter(labels)
    # Keep top N labels, group rest as "Other"
    top_labels = [l for l, _ in label_counts.most_common(max_legend)]
    plot_labels = [l if l in top_labels else "Other" for l in labels]

    unique_labels = sorted(set(plot_labels))
    # Move "Other" to end if present
    if "Other" in unique_labels:
        unique_labels.remove("Other")
        unique_labels.append("Other")

    # Color map
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
    if "Other" in color_map:
        color_map["Other"] = (0.8, 0.8, 0.8, 0.5)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot "Other" first (background)
    for label in unique_labels:
        mask = [l == label for l in plot_labels]
        pts = coords[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[color_map[label]],
            label=f"{label} ({label_counts.get(label, sum(1 for l in labels if l not in top_labels))})",
            s=12 if label != "Other" else 6,
            alpha=0.7 if label != "Other" else 0.3,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9, markerscale=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cluster_density(coords, labels, title, filename, figsize=(14, 10)):
    """Hex-bin density plot showing embedding concentration."""
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(
        coords[:, 0], coords[:, 1],
        gridsize=30, cmap="YlOrRd", mincnt=1,
    )
    cb = plt.colorbar(hb, ax=ax, label="Article count")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def compute_cluster_stats(coords, labels):
    """Compute per-topic centroid distances to show separation quality."""
    unique = list(set(labels))
    centroids = {}
    for label in unique:
        mask = [l == label for l in labels]
        pts = coords[mask]
        if len(pts) >= 5:  # Only topics with enough points
            centroids[label] = pts.mean(axis=0)

    # Compute avg intra-cluster distance vs inter-cluster distance
    print("\n  Cluster Separation Analysis:")
    print(f"  {'Topic':<25} {'Count':>6} {'Intra-dist':>12} {'Nearest':>25} {'Inter-dist':>12}")
    print("  " + "-" * 85)

    for label in sorted(centroids.keys()):
        mask = [l == label for l in labels]
        pts = coords[mask]
        centroid = centroids[label]
        intra = np.mean(np.linalg.norm(pts - centroid, axis=1))

        # Find nearest other cluster
        min_inter = float("inf")
        nearest = ""
        for other, other_c in centroids.items():
            if other == label:
                continue
            d = np.linalg.norm(centroid - other_c)
            if d < min_inter:
                min_inter = d
                nearest = other

        print(f"  {label:<25} {len(pts):>6} {intra:>12.2f} {nearest:>25} {min_inter:>12.2f}")


def main():
    print("Loading embeddings from database...")
    embeddings, topics, categories, sources = load_data()
    print(f"  Loaded {len(embeddings)} articles with embeddings")
    print(f"  Topics: {len(set(topics))} unique")
    print(f"  Categories: {len(set(categories))} unique")
    print(f"  Sources: {len(set(sources))} unique")

    # Run t-SNE once, reuse for all plots
    coords = run_tsne(embeddings)

    # Plot 1: By primary topic
    print("\nPlot 1: Articles colored by primary topic")
    plot_by_label(
        coords, topics,
        "Article Embedding Space — Colored by Primary Topic\n(t-SNE of 1024-dim mxbai-embed-large vectors)",
        "embeddings_by_topic.png",
    )

    # Plot 2: By category
    print("\nPlot 2: Articles colored by category")
    plot_by_label(
        coords, categories,
        "Article Embedding Space — Colored by Category\n(research / product / industry / open_source / opinion)",
        "embeddings_by_category.png",
    )

    # Plot 3: By source
    print("\nPlot 3: Articles colored by source")
    plot_by_label(
        coords, sources,
        "Article Embedding Space — Colored by Source",
        "embeddings_by_source.png",
    )

    # Plot 4: Density
    print("\nPlot 4: Embedding density")
    plot_cluster_density(
        coords, topics,
        "Article Embedding Density\n(highlights content concentration areas)",
        "embeddings_density.png",
    )

    # Cluster separation stats
    compute_cluster_stats(coords, topics)

    print("\n✅ Done! All plots saved to scripts/")


if __name__ == "__main__":
    main()

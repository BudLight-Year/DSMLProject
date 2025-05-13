"""
Demo for T-SNE visualization of vector space.

Add words to demo_words down below and run to save two semantic space images.

Display saved images side by side to see any interesting correlations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_model(model_path):
    """
    Load a Word2Vec model

    Args:
        model_path (str): Path to the model file

    Returns:
        Word2Vec: Loaded model
    """
    try:
        print(f"Loading model from {model_path}")
        model = Word2Vec.load(model_path)
        print(f"Vocabulary size: {len(model.wv.key_to_index):,} words")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def visualize_semantic_space(model, words, output_file=None, title="Word2Vec TSNE Visualization"):
    """
    Visualize semantic space for selected words using TSNE

    Args:
        model (Word2Vec): Loaded Word2Vec model
        words (list): List of words to visualize
        output_file (str): Path to save the visualization
        title (str): Title for the visualization
    """
    # Filter words to those that exist in the model
    valid_words = []
    for word in words:
        if word in model.wv:
            valid_words.append(word)
        else:
            print(f"Skipping '{word}' - not in model vocabulary")

    if not valid_words:
        print("No valid words to visualize")
        return

    print(f"Creating visualization for {len(valid_words)} words")

    # Get vectors for valid words
    vectors = [model.wv[word] for word in valid_words]
    vectors = np.array(vectors)

    # Apply TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, max(5, len(valid_words)-1)))
    vectors_2d = tsne.fit_transform(vectors)

    df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
    df['word'] = valid_words

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', data=df, alpha=0.7)

    for i, row in df.iterrows():
        plt.text(row['x']+0.02, row['y']+0.02, row['word'], fontsize=9)

    plt.title(title)
    plt.tight_layout()

    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")

    plt.show()


def compare_two_models(model_path1, model_path2, period1, period2,
                       words, output_dir="demo-visualizations"):
    """
    Create visualizations comparing two Word2Vec models

    Args:
        model_path1 (str): Path to the first model
        model_path2 (str): Path to the second model
        period1 (str): Name/period of the first model
        period2 (str): Name/period of the second model
        words (list): List of words to include in visualizations
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load both models
    print(f"Loading models for {period1} and {period2}...")
    model1 = load_model(model_path1)
    model2 = load_model(model_path2)

    if not model1 or not model2:
        print("Failed to load one or both models. Exiting.")
        return

    # Find common words between both models
    print("Finding words common to both models...")
    final_words = []
    for word in words:
        if word in model1.wv and word in model2.wv:
            final_words.append(word)
        else:
            if word not in model1.wv:
                print(f"'{word}' not in {period1} model")
            if word not in model2.wv:
                print(f"'{word}' not in {period2} model")

    print(f"Found {len(final_words)} words common to both models")

    if not final_words:
        print("No common words found. Exiting.")
        return

    # Create visualizations for each model
    for period, model in [(period1, model1), (period2, model2)]:
        output_file = os.path.join(output_dir, f"semantic_space_{period}.png")
        visualize_semantic_space(
            model,
            final_words,
            output_file=output_file,
            title=f"Semantic Space - {period}"
        )

    print(f"Visualizations saved to {output_dir}")


# Demo usage
if __name__ == "__main__":
    # Define model paths and periods
    model_path1 = "final_07_09_model/final_07_09.model"
    model_path2 = "final_24_model/final_2024_skipgram.model"
    period1 = "2007-2009"
    period2 = "2024"

    # Define words to visualize
    demo_words = [
        'president', 'economy', 'climate', 'health', 'social', 'media',
        'technology', 'internet', 'digital', 'data', 'security',
        'algorithm', 'cloud', 'mobile', 'application', 'ai',
        'news', 'sports', 'art', 'game', 'music', 'movie',

    ]

    # Run the two-model comparison
    compare_two_models(model_path1, model_path2, period1, period2, demo_words)

"""
Word2Vec Semantic Shift Analysis Tool

This script analyzes semantic changes across time periods using Word2Vec models.
It provides a comprehensive framework for comparing word embeddings trained on
different temporal corpora to identify and visualize semantic drift.

Key Functionality:
1. Loading multiple Word2Vec models representing different time periods
2. Identifying common and unique vocabulary across time periods
3. Analyzing semantic shift of specific words through time
4. Comparing word contexts across different periods
5. Visualizing semantic spaces using dimensionality reduction (t-SNE or PCA)
6. Calculating vector distances to quantify semantic change
7. Identifying words with the largest semantic shifts between periods
8. Comparing word-context relationships across time periods
9. Generating visualizations and saving analysis results

Command-line arguments allow for customization of input models, word lists,
context pairs, and output settings.

Usage:
python script_name.py --models period1:path1 period2:path2 --ouput-dir output_folder --words word1 word2 --contexts word1:context1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_models(model_paths):
    """
    Load multiple Word2Vec models

    Args:
        model_paths (dict): Dictionary mapping period names to model paths

    Returns:
        dict: Dictionary mapping period names to loaded models
    """
    models = {}
    for period, path in model_paths.items():
        try:
            print(f"Loading model for period '{period}' from {path}")
            models[period] = Word2Vec.load(path)
            print(
                f"  Vocabulary size: {len(models[period].wv.key_to_index):,} words")
        except Exception as e:
            print(f"Error loading model for period '{period}': {e}")

    return models


def common_vocabulary(models):
    """
    Find words that exist in all models

    Args:
        models (dict): Dictionary of loaded models

    Returns:
        list: List of words common to all models
    """
    vocabs = {period: set(model.wv.key_to_index.keys())
              for period, model in models.items()}
    common_words = set.intersection(*vocabs.values())
    print(f"Found {len(common_words):,} words common to all periods")

    return list(common_words)


def unique_vocabulary(models):
    """
    Find words unique to each time period

    Args:
        models (dict): Dictionary of loaded models

    Returns:
        dict: Dictionary mapping period names to unique words
    """
    vocabs = {period: set(model.wv.key_to_index.keys())
              for period, model in models.items()}
    unique_words = {}

    for period, vocab in vocabs.items():
        other_vocabs = set.union(
            *[v for p, v in vocabs.items() if p != period])
        unique_words[period] = vocab - other_vocabs
        print(f"Period '{period}': {len(unique_words[period]):,} unique words")

    return unique_words


def semantic_shift(models, words, top_n=10):
    """
    Analyze semantic shift of words between time periods

    Args:
        models (dict): Dictionary of loaded models
        words (list): List of words to analyze
        top_n (int): Number of similar words to show

    Returns:
        dict: Dictionary with semantic shift analysis
    """
    results = {}

    for word in words:
        print(f"\nSemantic shift analysis for '{word}':")
        word_results = {}

        # Check if word exists in all models
        exists_in_all = all(word in model.wv for model in models.values())
        if not exists_in_all:
            print(f"  '{word}' does not exist in all time periods")
            missing_in = [period for period,
                          model in models.items() if word not in model.wv]
            print(f"  Missing in: {', '.join(missing_in)}")
            continue

        # Get similar words for each period
        for period, model in models.items():
            try:
                similar = model.wv.most_similar(word, topn=top_n)
                word_results[period] = similar

                print(f"\n  {period} similar words:")
                for similar_word, similarity in similar:
                    print(f"    {similar_word}: {similarity:.4f}")
            except KeyError:
                print(f"  '{word}' not in vocabulary for period '{period}'")

        results[word] = word_results

    return results


def contextual_comparison(models, context_pairs, window=5):
    """
    Compare contexts in which terms appear across time periods

    Args:
        models (dict): Dictionary of loaded models
        context_pairs (list): List of (word, context) pairs to analyze
        window (int): Number of similar words to show

    Returns:
        dict: Dictionary with contextual analysis
    """
    results = {}

    for word, context in context_pairs:
        print(f"\nAnalyzing '{word}' in context of '{context}':")
        pair_results = {}

        for period, model in models.items():
            if word not in model.wv or context not in model.wv:
                print(f"  Word or context missing in period '{period}'")
                continue

            similarity = model.wv.similarity(word, context)

            # Find words similar to both the word and the context
            try:
                common_similar = []
                word_similar = set(
                    w for w, _ in model.wv.most_similar(word, topn=50))
                context_similar = set(
                    w for w, _ in model.wv.most_similar(context, topn=50))
                common_words = word_similar.intersection(context_similar)

                # Sort by average similarity
                common_similar = [(w, (model.wv.similarity(word, w) + model.wv.similarity(context, w))/2)
                                  for w in common_words]
                common_similar.sort(key=lambda x: x[1], reverse=True)

                pair_results[period] = {
                    'similarity': similarity,
                    'common_similar': common_similar[:window]
                }

                print(f"\n  {period}:")
                print(f"    Similarity: {similarity:.4f}")
                print(f"    Words similar to both '{word}' and '{context}':")
                for similar_word, sim in common_similar[:window]:
                    print(f"      {similar_word}: {sim:.4f}")

            except KeyError as e:
                print(f"  Error in period '{period}': {e}")

        results[(word, context)] = pair_results

    return results


def visualize_semantic_space(models, words, output_dir=None, method='tsne'):
    """
    Visualize semantic space for selected words across time periods

    Args:
        models (dict): Dictionary of loaded models
        words (list): List of words to visualize
        output_dir (str): Directory to save visualizations
        method (str): Dimensionality reduction method ('tsne' or 'pca')
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter words to those that exist in all models
    valid_words = []
    for word in words:
        if all(word in model.wv for model in models.values()):
            valid_words.append(word)
        else:
            print(f"Skipping '{word}' - not in all models")

    if not valid_words:
        print("No common words to visualize")
        return

    # Create visualizations
    for period, model in models.items():
        print(f"Creating visualization for period '{period}'")

        vectors = [model.wv[word] for word in valid_words]
        vectors = np.array(vectors)

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)

        vectors_2d = reducer.fit_transform(vectors)

        df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
        df['word'] = valid_words

        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', data=df, alpha=0.7)

        for i, row in df.iterrows():
            plt.text(row['x']+0.02, row['y']+0.02, row['word'], fontsize=9)

        plt.title(f"Semantic space - {period} ({method.upper()})")
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(
                output_dir, f"semantic_space_{period}_{method}.png"))
            plt.close()
        else:
            plt.show()


def compare_word_vectors(models, words):
    """
    Compare word vectors across time periods by calculating cosine distances

    Args:
        models (dict): Dictionary of loaded models
        words (list): List of words to compare

    Returns:
        dict: Dictionary with vector comparisons
    """
    results = {}
    periods = list(models.keys())

    for word in words:
        print(f"\nVector comparison for '{word}':")

        # Check if word exists in all models
        exists_in_all = all(word in model.wv for model in models.values())
        if not exists_in_all:
            print(f"  '{word}' does not exist in all time periods")
            missing_in = [period for period,
                          model in models.items() if word not in model.wv]
            print(f"  Missing in: {', '.join(missing_in)}")
            continue

        vectors = {period: models[period].wv[word] for period in periods}

        # Compare vectors between all period pairs
        comparisons = {}
        for i, period1 in enumerate(periods):
            for period2 in periods[i+1:]:
                vec1 = vectors[period1]
                vec2 = vectors[period2]

                # Calculate cosine similarity (dot product of normalized vectors)
                similarity = np.dot(vec1, vec2) / \
                    (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                distance = 1 - similarity

                comparisons[(period1, period2)] = distance
                print(
                    f"  Distance between {period1} and {period2}: {distance:.4f}")

        results[word] = comparisons

    return results


def top_changed_words(models, num_words=100, min_freq=50):
    """
    Find words with the largest semantic shift between time periods

    Args:
        models (dict): Dictionary of loaded models
        num_words (int): Number of top changed words to return
        min_freq (int): Minimum frequency for words to consider

    Returns:
        list: List of (word, distance) pairs sorted by semantic change
    """
    if len(models) != 2:
        print("This analysis requires exactly 2 time periods")
        return []

    periods = list(models.keys())
    model1, model2 = models[periods[0]], models[periods[1]]

    # Find common vocabulary
    vocab1 = set(model1.wv.key_to_index.keys())
    vocab2 = set(model2.wv.key_to_index.keys())
    common_vocab = vocab1.intersection(vocab2)

    print(f"Comparing {len(common_vocab):,} common words between {periods[0]} and {periods[1]}")

    # Calculate semantic distance for each word
    distances = []
    for word in tqdm(common_vocab, desc="Calculating semantic shifts"):
        vec1 = model1.wv[word]
        vec2 = model2.wv[word]

        # Calculate cosine similarity and convert to distance
        similarity = np.dot(vec1, vec2) / \
            (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        distance = 1 - similarity

        distances.append((word, distance))

    distances.sort(key=lambda x: x[1], reverse=True)

    top_words = distances[:num_words]

    print("\nTop words with largest semantic shift:")
    for word, distance in top_words[:20]:
        print(f"  {word}: {distance:.4f}")

    return top_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare Word2Vec Models Across Time Periods')
    parser.add_argument('--models', '-m', required=True, nargs='+',
                        help='Paths to Word2Vec models (format: period:path)')
    parser.add_argument('--output-dir', '-o', default='comparison_results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--words', '-w', nargs='+', default=[
                        'president', 'economy', 'climate',
                        'health', 'social', 'media', 'phone', 'computer',
                        'technology', 'internet', 'software', 'hardware',
                        'digital', 'data', 'network', 'security', 'programming',
                        'algorithm', 'website', 'code', 'server', 'cloud',
                        'mobile', 'application', 'device', 'system', 'interface',
                        'crime', 'judge', 'apple', 'microsoft', 'gpu',
                        'graphics', 'ai', 'song', 'movie',
                        'fracking', 'slavery', 'religion', 'goo', 'burglar',
                        'mcdonalds', 'law', 'music', 'game', 'twitch',
                        'twitter', 'museum', 'cracker', 'military',
                        'war', 'restaurant', 'immigrants', 'ice', 'illegal',
                        'news', 'sports', 'vegas', 'prickly', 'streaming',
                        'youtube', 'kids', 'children', 'art', 'gaming',
                        'cactus', 'net', 'neutrality', 'nasa', 'space', 'satellite',
                        'artificial', 'intelligence', 'vr', 'virtual', 'reality',
                        'man', 'king', 'woman', 'queen', 'yas'],
                        help='Words to analyze for semantic shift')
    parser.add_argument('--contexts', '-c', nargs='+', default=[
                        'social:media', 'climate:change', 'health:care', 'video:game',
                        'kid:crime', 'programming:language', 'mobile:phone', 'cloud:computing',
                        'microsoft:software', 'apple:software', 'prickly:pear',
                        'net:neutrality', 'microsoft:hardware', 'apple:hardware',
                        'artificial:intelligence', 'virtual:reality'],
                        help='Word-context pairs to analyze (format: word:context)')
    parser.add_argument('--top-changed', '-t', type=int, default=100,
                        help='Number of top changed words to find')
    parser.add_argument('--visualization', '-v', choices=['tsne', 'pca', 'none'],
                        default='tsne', help='Visualization method')

    args = parser.parse_args()

    model_paths = {}
    for model_arg in args.models:
        parts = model_arg.split(':')
        if len(parts) != 2:
            print(f"Invalid model format: {model_arg}. Use 'period:path'")
            continue
        period, path = parts
        model_paths[period] = path

    models = load_models(model_paths)

    if not models:
        print("No models loaded. Exiting.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    common_words = common_vocabulary(models)
    unique_words = unique_vocabulary(models)

    # Save unique words to files
    for period, words in unique_words.items():
        with open(os.path.join(args.output_dir, f"unique_words_{period}.txt"), 'w', encoding='utf-8') as f:
            # Limit to first 1000 words
            for word in sorted(list(words)[:1000]):
                f.write(f"{word}\n")

    # Process context pairs
    context_pairs = []
    for context_arg in args.contexts:
        parts = context_arg.split(':')
        if len(parts) != 2:
            print(f"Invalid context format: {context_arg}. Use 'word:context'")
            continue
        word, context = parts
        context_pairs.append((word, context))

    # Perform analyses
    semantic_results = semantic_shift(models, args.words)
    context_results = contextual_comparison(models, context_pairs)
    vector_comparisons = compare_word_vectors(models, args.words)

    # Find top changed words and save
    if len(models) == 2:
        changed_words = top_changed_words(models, num_words=args.top_changed)
        with open(os.path.join(args.output_dir, "top_changed_words.txt"), 'w', encoding='utf-8') as f:
            for word, distance in changed_words:
                f.write(f"{word}\t{distance:.6f}\n")

    # Create visualizations
    if args.visualization != 'none':
        visualize_semantic_space(
            models, args.words,
            output_dir=args.output_dir,
            method=args.visualization
        )

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

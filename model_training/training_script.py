"""
Functions and script to load pre-processed data and
train a Word to Vector model from the gensim library.
Runs a simple evaluation after training.

Some code in this module was adapted from code provided by Claude (Anthropic).
"""
import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from tqdm import tqdm
import logging
import argparse
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_processed_files(input_dir, file_pattern=None, sample_size=None, year_filter=None, verbose=True):
    """
    Load preprocessed files and prepare them for Word2Vec training.

    Args:
        input_dir (str): Directory containing processed files
        file_pattern (str, optional): Pattern to match specific files (e.g., '*.csv')
        sample_size (int, optional): Number of files to sample (for testing)
        year_filter (list, optional): List of years to include (e.g., [2007, 2008, 2009])
        verbose (bool): Whether to show detailed loading information

    Returns:
        list: List of token lists ready for Word2Vec training
    """
    if file_pattern is None:
        file_pattern = '*.csv'

    pattern = os.path.join(input_dir, file_pattern)
    all_files = sorted(glob.glob(pattern))

    if verbose:
        print(f"Found {len(all_files)} files matching pattern: {pattern}")

    if sample_size and sample_size < len(all_files):
        if verbose:
            print(f"Sampling {sample_size} files for training...")
        all_files = all_files[:sample_size]

    if year_filter:
        filtered_files = []
        for file in all_files:
            for year in year_filter:
                if f"{year}" in os.path.basename(file):
                    filtered_files.append(file)
                    break

        if verbose:
            print(f"Filtered to {len(filtered_files)} files from years: {year_filter}")
        all_files = filtered_files

    token_lists = []
    total_documents = 0

    if verbose:
        print("Loading and preparing files for Word2Vec training...")

    for file in tqdm(all_files, desc="Loading files", disable=not verbose):
        try:
            df = pd.read_csv(file, header=None, names=['tokens', 'entities'], quoting=1)

            file_tokens = df['tokens'].apply(lambda x: str(x).split())

            token_lists.extend(file_tokens)
            total_documents += len(file_tokens)

        except Exception as e:
            print(f"Error loading file {file}: {e}")

    if verbose:
        print(f"Loaded {total_documents:,} documents with {len(token_lists):,} token lists")

    return token_lists


def train_word2vec(token_lists, vector_size=300, window=5, min_count=5, workers=None, 
                   sg=1, epochs=5, output_path=None, verbose=True):
    """
    Train a Word2Vec model on the preprocessed token lists.

    Args:
        token_lists (list): List of token lists
        vector_size (int): Dimensionality of word vectors
        window (int): Maximum distance for context words
        min_count (int): Minimum word frequency
        workers (int, optional): Number of CPU cores to use
        sg (int): Training algorithm (1 for skip-gram, 0 for CBOW)
        epochs (int): Number of training passes
        output_path (str, optional): Path to save the model
        verbose (bool): Whether to show detailed training information

    Returns:
        Word2Vec: Trained Word2Vec model
    """
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    if verbose:
        print(f"Training Word2Vec model with {workers} workers...")
        print(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, sg={sg}, epochs={epochs}")

    start_time = time.time()

    # Train the Word to Vector model
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # 1 for skip-gram, 0 for CBOW
        epochs=epochs
    )

    training_time = time.time() - start_time

    if verbose:
        vocab_size = len(model.wv.key_to_index)
        print(f"Model trained in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Vocabulary size: {vocab_size:,} unique words")

    if output_path:
        model.save(output_path)
        if verbose:
            print(f"Model saved to: {output_path}")

    return model


def evaluate_model(model, num_words=20, verbose=True):
    """
    Perform basic evaluation of the trained Word2Vec model.

    Args:
        model (Word2Vec): Trained Word2Vec model
        num_words (int): Number of words to use for similarity examples
        verbose (bool): Whether to show detailed results
    """
    common_words = list(model.wv.key_to_index.keys())[:100]
    test_words = common_words[:num_words]

    if verbose:
        print("\nWord2Vec Model Evaluation:")
        print("==========================")

        for word in test_words:
            try:
                similar_words = model.wv.most_similar(word, topn=5)
                print(f"\nWords most similar to '{word}':")
                for similar_word, similarity in similar_words:
                    print(f"  {similar_word}: {similarity:.4f}")
            except KeyError:
                print(f"Word '{word}' not in vocabulary")

    analogy_examples = [
        ('man', 'woman', 'king'),
        ('paris', 'france', 'rome'),
        ('good', 'better', 'bad')
    ]

    if verbose:
        print("\nWord Analogies:")
        print("==============")

        for word1, word2, word3 in analogy_examples:
            try:
                result = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)
                print(f"{word1} is to {word2} as {word3} is to {result[0][0]} (similarity: {result[0][1]:.4f})")
            except KeyError:
                print(f"Analogy example with {word1}, {word2}, {word3} failed (word not in vocabulary)")


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description='Train Word2Vec on preprocessed NLP data')
    parser.add_argument('--input-dir', '-i', required=True, help='Directory containing processed files')
    parser.add_argument('--output', '-o', default='word2vec_model.model', help='Path to save the model')
    parser.add_argument('--file-pattern', '-p', default='*.csv', help='Pattern to match specific files')
    parser.add_argument('--vector-size', type=int, default=300, help='Dimensionality of word vectors')
    parser.add_argument('--window', type=int, default=5, help='Maximum distance for context words')
    parser.add_argument('--min-count', type=int, default=5, help='Minimum word frequency')
    parser.add_argument('--workers', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1], 
                        help='Training algorithm (1 for skip-gram, 0 for CBOW)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training passes')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of files to sample (for testing)')
    parser.add_argument('--years', type=int, nargs='+', help='Filter files by years')
    parser.add_argument('--no-eval', action='store_true', help='Skip model evaluation')

    args = parser.parse_args()

    # Load data
    token_lists = load_processed_files(
        args.input_dir,
        file_pattern=args.file_pattern,
        sample_size=args.sample_size,
        year_filter=args.years
    )

    # Train model
    model = train_word2vec(
        token_lists,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        epochs=args.epochs,
        output_path=args.output
    )

    # Run simple evaluation
    if not args.no_eval:
        evaluate_model(model)

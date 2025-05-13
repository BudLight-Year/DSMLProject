"""
Functions and script to load pre-processed data and
train a Word to Vector model from the gensim library.
Runs a simple evaluation after training.

Modified to use optimal parameters based on grid search results.
Some code in this module was adapted from code provided by Claude (Anthropic).
"""
import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
import logging
import argparse
import time

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LossCallback(CallbackAny2Vec):
    """Callback to track loss during Word2Vec training"""

    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch} starting...")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            self.loss = loss
        else:
            current_loss = loss - self.previous_loss
            self.losses.append(current_loss)
            print(f"Loss in epoch {self.epoch}: {current_loss}")
        self.previous_loss = loss


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
            print(
                f"Filtered to {len(filtered_files)} files from years: {year_filter}")
        all_files = filtered_files

    token_lists = []
    total_documents = 0

    if verbose:
        print("Loading and preparing files for Word2Vec training...")

    for file in tqdm(all_files, desc="Loading files", disable=not verbose):
        try:
            df = pd.read_csv(file, header=None, names=[
                             'tokens', 'entities'], quoting=1)

            file_tokens = df['tokens'].apply(lambda x: str(x).split())

            token_lists.extend(file_tokens)
            total_documents += len(file_tokens)

        except Exception as e:
            print(f"Error loading file {file}: {e}")

    if verbose:
        print(
            f"Loaded {total_documents:,} documents with {len(token_lists):,} token lists")

    return token_lists


def train_word2vec(token_lists, vector_size=300, window=5, min_count=5, workers=None,
                   sg=1, hs=0, negative=5, ns_exponent=0.75, alpha=0.025, min_alpha=0.0001,
                   sample=0.001, epochs=5, output_path=None, verbose=True, compute_loss=False):
    """
    Train a Word2Vec model on the preprocessed token lists with optimal parameters.

    Args:
        token_lists (list): List of token lists
        vector_size (int): Dimensionality of word vectors
        window (int): Maximum distance for context words
        min_count (int): Minimum word frequency
        workers (int, optional): Number of CPU cores to use
        sg (int): Training algorithm (1 for skip-gram, 0 for CBOW)
        hs (int): Use hierarchical softmax (0 for negative sampling)
        negative (int): Number of negative samples
        ns_exponent (float): Negative sampling distribution exponent
        alpha (float): Initial learning rate
        min_alpha (float): Final learning rate
        sample (float): Subsampling threshold for frequent words
        epochs (int): Number of training passes
        output_path (str, optional): Path to save the model
        verbose (bool): Whether to show detailed training information
        compute_loss (bool): Whether to compute and display loss during training

    Returns:
        Word2Vec: Trained Word2Vec model
    """
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    if verbose:
        print(f"Training Word2Vec model with {workers} workers...")
        print(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, "
              f"sg={sg}, hs={hs}, negative={negative}, ns_exponent={ns_exponent}, "
              f"alpha={alpha}, min_alpha={min_alpha}, sample={sample}, epochs={epochs}")

    start_time = time.time()

    # Setup loss callback if requested
    callbacks = []
    if compute_loss:
        loss_callback = LossCallback()
        callbacks.append(loss_callback)

    # Train the Word to Vector model
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # 1 for skip-gram, 0 for CBOW
        hs=hs,  # 0 for negative sampling, 1 for hierarchical softmax
        negative=negative,  # Number of negative samples
        ns_exponent=ns_exponent,  # Negative sampling distribution exponent
        alpha=alpha,  # Initial learning rate
        min_alpha=min_alpha,  # Final learning rate
        sample=sample,  # Subsampling threshold for frequent words
        epochs=epochs,
        compute_loss=compute_loss,
        callbacks=callbacks
    )

    training_time = time.time() - start_time

    if verbose:
        vocab_size = len(model.wv.key_to_index)
        print(
            f"Model trained in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Vocabulary size: {vocab_size:,} unique words")

        if compute_loss and callbacks:
            print(f"Final training loss: {loss_callback.previous_loss}")

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
        ('good', 'better', 'bad'),
        ('car', 'cars', 'child'),
        ('berlin', 'germany', 'tokyo')
    ]

    if verbose:
        print("\nWord Analogies:")
        print("==============")

        for word1, word2, word3 in analogy_examples:
            try:
                result = model.wv.most_similar(
                    positive=[word2, word3], negative=[word1], topn=1)
                print(
                    f"{word1} is to {word2} as {word3} is to {result[0][0]} (similarity: {result[0][1]:.4f})")
            except KeyError:
                print(
                    f"Analogy example with {word1}, {word2}, {word3} failed (word not in vocabulary)")


def test_analogies(model, verbose=True):
    """
    Test model on standard analogy categories.

    Args:
        model (Word2Vec): Trained Word2Vec model
        verbose (bool): Whether to show detailed results

    Returns:
        dict: Results by category and overall performance
    """
    categories = {
        'gender': [
            ('man', 'woman', 'king', 'queen'),
            ('man', 'woman', 'husband', 'wife'),
            ('man', 'woman', 'actor', 'actress'),
            ('man', 'woman', 'father', 'mother'),
            ('boy', 'girl', 'brother', 'sister'),
            ('nephew', 'niece', 'uncle', 'aunt')
        ],
        'capital-country': [
            ('paris', 'france', 'rome', 'italy'),
            ('berlin', 'germany', 'tokyo', 'japan'),
            ('london', 'england', 'madrid', 'spain'),
            ('washington', 'usa', 'moscow', 'russia'),
            ('beijing', 'china', 'seoul', 'korea'),
            ('ottawa', 'canada', 'canberra', 'australia')
        ],
        'comparative-superlative': [
            ('good', 'best', 'bad', 'worst'),
            ('big', 'biggest', 'small', 'smallest'),
            ('fast', 'fastest', 'slow', 'slowest'),
            ('happy', 'happiest', 'sad', 'saddest'),
            ('hard', 'hardest', 'easy', 'easiest'),
            ('long', 'longest', 'short', 'shortest')
        ],
        'plural': [
            ('car', 'cars', 'child', 'children'),
            ('dog', 'dogs', 'mouse', 'mice'),
            ('house', 'houses', 'person', 'people'),
            ('cat', 'cats', 'foot', 'feet'),
            ('bird', 'birds', 'tooth', 'teeth'),
            ('book', 'books', 'goose', 'geese')
        ]
    }

    results = {}

    for category, analogies in categories.items():
        if verbose:
            print(f"\n=== Testing Category: {category} ===")

        category_results = {
            'total': len(analogies),
            'in_vocab': 0,
            'successful': 0,
            'ranks': []
        }

        for a, b, c, expected in analogies:
            # Check if all words are in vocabulary
            if all(w in model.wv for w in [a, b, c, expected]):
                category_results['in_vocab'] += 1

                # Try to predict the fourth word
                try:
                    predictions = model.wv.most_similar(
                        positive=[c, b], negative=[a], topn=100)
                    predicted_words = [word for word, _ in predictions]

                    if expected in predicted_words:
                        rank = predicted_words.index(expected) + 1
                        category_results['ranks'].append(rank)

                        if rank == 1:
                            category_results['successful'] += 1
                            if verbose:
                                print(f"✓ {a}:{b} :: {c}:{expected}")
                                print(
                                    f"  Predicted: {predicted_words[0]} (score: {predictions[0][1]:.4f})")
                        else:
                            if verbose:
                                print(f"✗ {a}:{b} :: {c}:{expected}")
                                print(
                                    f"  Predicted: {predicted_words[0]} (score: {predictions[0][1]:.4f})")
                                print(
                                    f"  Expected '{expected}' found at rank {rank}")
                    else:
                        # Word not in top 100
                        category_results['ranks'].append(101)
                        if verbose:
                            print(f"✗ {a}:{b} :: {c}:{expected}")
                            print(
                                f"  Predicted: {predicted_words[0]} (score: {predictions[0][1]:.4f})")
                            print(
                                f"  Expected '{expected}' not found in top 100")
                except Exception as e:
                    # Error during prediction
                    if verbose:
                        print(
                            f"Error with analogy {a}:{b} :: {c}:{expected}: {e}")
            else:
                missing = [w for w in [a, b, c, expected] if w not in model.wv]
                if verbose:
                    print(
                        f"Word '{missing[0]}' not in vocabulary, skipping analogy")

        # Calculate metrics
        if category_results['in_vocab'] > 0:
            category_results['accuracy_vocab'] = category_results['successful'] / \
                category_results['in_vocab']
            category_results['accuracy_total'] = category_results['successful'] / \
                category_results['total']

            if category_results['ranks']:
                category_results['mean_rank'] = np.mean(
                    category_results['ranks'])
                category_results['median_rank'] = np.median(
                    category_results['ranks'])
            else:
                category_results['mean_rank'] = 0
                category_results['median_rank'] = 0
        else:
            category_results['accuracy_vocab'] = 0
            category_results['accuracy_total'] = 0
            category_results['mean_rank'] = 0
            category_results['median_rank'] = 0

        if verbose:
            print(f"\nResults for category '{category}':")
            print(f"Total analogies: {category_results['total']}")
            print(
                f"Analogies with all words in vocabulary: {category_results['in_vocab']}")
            print(f"Successful predictions: {category_results['successful']}")
            print(
                f"Accuracy (vocab): {category_results['accuracy_vocab']:.2%}")
            print(
                f"Accuracy (total): {category_results['accuracy_total']:.2%}")
            if category_results['ranks']:
                print(
                    f"Mean rank of expected word: {category_results['mean_rank']:.2f}")
                print(
                    f"Median rank of expected word: {category_results['median_rank']:.2f}")

        results[category] = category_results

    # Calculate overall metrics
    total_analogies = sum(r['total'] for r in results.values())
    total_in_vocab = sum(r['in_vocab'] for r in results.values())
    total_successful = sum(r['successful'] for r in results.values())
    all_ranks = [rank for r in results.values()
                 for rank in r['ranks'] if rank <= 100]

    results['overall'] = {
        'total': total_analogies,
        'in_vocab': total_in_vocab,
        'successful': total_successful,
        'accuracy_vocab': total_successful / total_in_vocab if total_in_vocab > 0 else 0,
        'accuracy_total': total_successful / total_analogies if total_analogies > 0 else 0,
        'mean_rank': np.mean(all_ranks) if all_ranks else 0,
        'median_rank': np.median(all_ranks) if all_ranks else 0
    }

    if verbose:
        print("\n=== Overall Analogy Test Results ===")
        print(f"Total analogies tested: {total_analogies}")
        print(f"Analogies with all words in vocabulary: {total_in_vocab}")
        print(f"Successful predictions: {total_successful}")
        print(f"Accuracy (vocab): {results['overall']['accuracy_vocab']:.2%}")
        print(f"Accuracy (total): {results['overall']['accuracy_total']:.2%}")
        if all_ranks:
            print(
                f"Mean rank of expected word: {results['overall']['mean_rank']:.2f}")
            print(
                f"Median rank of expected word: {results['overall']['median_rank']:.2f}")

    return results


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(
        description='Train Word2Vec on preprocessed NLP data with optimized parameters')
    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing processed files')
    parser.add_argument(
        '--output', '-o', default='word2vec_model.model', help='Path to save the model')
    parser.add_argument('--file-pattern', '-p', default='*.csv',
                        help='Pattern to match specific files')

    # Model parameters (set to optimal defaults based on grid search)
    parser.add_argument('--vector-size', type=int, default=300,
                        help='Dimensionality of word vectors')
    parser.add_argument('--window', type=int, default=5,
                        help='Maximum distance for context words')
    parser.add_argument('--min-count', type=int, default=5,
                        help='Minimum word frequency')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of CPU cores to use')
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1],
                        help='Training algorithm (1 for skip-gram, 0 for CBOW)')
    parser.add_argument('--hs', type=int, default=0, choices=[0, 1],
                        help='Use hierarchical softmax (0 for negative sampling)')
    parser.add_argument('--negative', type=int, default=5,
                        help='Number of negative samples')
    parser.add_argument('--ns-exponent', type=float, default=0.75,
                        help='Negative sampling distribution exponent')
    parser.add_argument('--alpha', type=float, default=0.025,
                        help='Initial learning rate')
    parser.add_argument('--min-alpha', type=float,
                        default=0.0001, help='Final learning rate')
    parser.add_argument('--sample', type=float, default=0.001,
                        help='Subsampling threshold for frequent words')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training passes')

    # Other arguments
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of files to sample (for testing)')
    parser.add_argument('--years', type=int, nargs='+',
                        help='Filter files by years')
    parser.add_argument('--compute-loss', action='store_true',
                        help='Compute and display loss during training')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip model evaluation')
    parser.add_argument('--analogy-test', action='store_true',
                        help='Run full analogy tests')

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
        hs=args.hs,
        negative=args.negative,
        ns_exponent=args.ns_exponent,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        sample=args.sample,
        epochs=args.epochs,
        output_path=args.output,
        compute_loss=args.compute_loss
    )

    # Run simple evaluation
    if not args.no_eval:
        evaluate_model(model)

    # Run analogy tests if requested
    if args.analogy_test:
        test_analogies(model)

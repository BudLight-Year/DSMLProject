"""
Word2Vec Grid Search Optimization Framework

This script implements a grid search approach to optimize Word2Vec embeddings with linguistic
analogy evaluation. It trains multiple Word2Vec models with different hyperparameter 
combinations, evaluates their performance on standard linguistic analogy tests 
(gender, capital-country, comparative-superlative, plural), and identifies the best configuration.

Key capabilities:
- Loading and processing tokenized text from CSV files
- Efficiently testing multiple hyperparameter combinations with result tracking
- Evaluating model quality using vector algebra on word analogies (A:B::C:D)
- Saving interim results to enable resuming interrupted parameter searches
- Skipping previously tested parameter combinations to avoid redundant work

Usage:
    python grid_search.py
"""

import os
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import glob
import csv
import time
from datetime import datetime
from tqdm import tqdm
from itertools import product
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LossTracker(CallbackAny2Vec):
    """
    Track and display loss during Word2Vec model training.

    This callback prints the loss after each epoch and keeps track
    of training time per epoch.
    """

    def __init__(self):
        """Initialize the loss tracker."""
        self.epoch = 0
        self.losses = []
        self.start_time = None

    def on_epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            model: The Word2Vec model being trained
        """
        self.epoch += 1
        self.start_time = time.time()
        print(f'Epoch {self.epoch} starting...')

    def on_epoch_end(self, model):
        """
        Called at the end of each epoch to calculate and display loss.

        Args:
            model: The Word2Vec model being trained
        """
        epoch_time = time.time() - self.start_time
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            previous_loss = 0
        else:
            previous_loss = self.losses[-1] if self.losses else 0
        current_loss = loss - previous_loss
        self.losses.append(current_loss)
        print(
            f'Epoch {self.epoch}: Loss = {current_loss:.4f} (Time: {epoch_time:.2f}s)')


def load_corpus_from_csv_files(directory, file_pattern="*.csv", max_files=None):
    """
    Load tokenized sentences from processed CSV files.

    Args:
        directory: Directory containing CSV files
        file_pattern: Pattern to match CSV files
        max_files: Maximum number of files to load (None for all)

    Returns:
        List of tokenized sentences
    """
    sentences = []
    files = sorted(glob.glob(os.path.join(directory, file_pattern)))

    if max_files:
        files = files[:max_files]

    print(f"Loading corpus from {len(files)} files in {directory}")

    for file_path in tqdm(files, desc="Loading corpus files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if row and len(row) > 0:
                        tokens = row[0].split()
                        if tokens:
                            sentences.append(tokens)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    print(f"Loaded {len(sentences)} sentences from {len(files)} files")
    return sentences


def test_analogies(model, categories=None):
    """
    Test Word2Vec model on analogy tasks.

    Args:
        model: Trained Word2Vec model
        categories: Dictionary of analogy categories and test cases
                   (None to use default categories)

    Returns:
        Dictionary containing test results by category
    """
    # Define standard linguistic analogy tests using the pattern A:B :: C:D
    # Tests whether model can predict D given A, B, and C (king - man + woman ≈ queen)
    if categories is None:
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
        category_results = {
            'total': len(analogies),
            'in_vocab': 0,
            'successful': 0,
            'ranks': []
        }

        # For each analogy tuple (A:B::C:D), test if the model correctly predicts D
        # using the vector arithmetic: vector(B) - vector(A) + vector(C) ≈ vector(D)
        for a, b, c, expected in analogies:
            if all(w in model.wv for w in [a, b, c, expected]):
                category_results['in_vocab'] += 1

                try:
                    predictions = model.wv.most_similar(
                        positive=[c, b], negative=[a], topn=100)
                    predicted_words = [word for word, _ in predictions]

                    if expected in predicted_words:
                        rank = predicted_words.index(expected) + 1
                        category_results['ranks'].append(rank)

                        if rank == 1:
                            category_results['successful'] += 1
                    else:
                        category_results['ranks'].append(101)
                except:
                    pass

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

        results[category] = category_results

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

    return results


def train_and_evaluate(params, sentences, output_dir=None):
    """
    Train a Word2Vec model with given parameters and evaluate its performance.

    Args:
        params: Dictionary of Word2Vec parameters
        sentences: List of tokenized sentences
        output_dir: Directory to save the model (None to skip saving)

    Returns:
        Dictionary containing model parameters and evaluation results
    """
    # Core model training and evaluation function - builds Word2Vec model with specified
    # hyperparameters and evaluates it on linguistic analogy tests
    try:
        model_name = f"w2v_vs{params['vector_size']}_w{params['window']}_e{params['epochs']}_sg{params['sg']}"
        print(f"\nTraining model: {model_name}")

        loss_tracker = LossTracker()

        start_time = time.time()

        model = Word2Vec(
            sentences=sentences,
            vector_size=params['vector_size'],
            window=params['window'],
            min_count=params['min_count'],
            sg=params['sg'],
            hs=params['hs'],
            negative=params['negative'],
            ns_exponent=params['ns_exponent'],
            alpha=params['alpha'],
            min_alpha=params['min_alpha'],
            sample=params['sample'],
            epochs=params['epochs'],
            workers=params['workers'],
            compute_loss=True,
            callbacks=[loss_tracker]
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        vocab_size = len(model.wv.index_to_key)
        print(f"Vocabulary size: {vocab_size}")

        print("Evaluating model on analogies...")
        analogy_results = test_analogies(model)

        overall_accuracy = analogy_results['overall']['accuracy_vocab']
        print(f"Overall accuracy: {overall_accuracy:.4f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            filename = f"{model_name}_acc{overall_accuracy:.4f}.model"
            model_path = os.path.join(output_dir, filename)

            model.save(model_path)
            print(f"Model saved to {model_path}")

        result = params.copy()
        result.update({
            'vocab_size': vocab_size,
            'training_time': training_time,
            'final_loss': loss_tracker.losses[-1] if loss_tracker.losses else None,
            'overall_accuracy': overall_accuracy,
            'mean_rank': analogy_results['overall']['mean_rank'],
            'median_rank': analogy_results['overall']['median_rank']
        })

        for category, category_results in analogy_results.items():
            if category != 'overall':
                result[f'{category}_accuracy'] = category_results['accuracy_vocab']
                result[f'{category}_mean_rank'] = category_results['mean_rank']

        return result

    except Exception as e:
        print(f"Error in train_and_evaluate: {e}")
        import traceback
        traceback.print_exc()

        result = params.copy()
        result.update({
            'vocab_size': 0,
            'training_time': 0,
            'final_loss': None,
            'overall_accuracy': 0,
            'mean_rank': 0,
            'median_rank': 0,
            'error': str(e)
        })
        return result


def load_previously_tested_params(results_csv):
    """
    Load parameter combinations that have already been tested.

    Args:
        results_csv: Path to CSV file containing previous results

    Returns:
        List of parameter dictionaries that have been tested
    """
    if not os.path.exists(results_csv):
        print(
            f"Previous results file {results_csv} not found. Starting fresh.")
        return []

    try:
        df = pd.read_csv(results_csv)

        param_columns = ['vector_size', 'window', 'min_count', 'sg', 'hs',
                         'negative', 'ns_exponent', 'alpha', 'min_alpha',
                         'sample', 'epochs', 'workers']

        param_columns = [col for col in param_columns if col in df.columns]

        tested_params = df[param_columns].to_dict('records')

        print(
            f"Loaded {len(tested_params)} previously tested parameter combinations from {results_csv}")
        return tested_params

    except Exception as e:
        print(f"Error loading previous results: {e}")
        return []


def is_already_tested(params, tested_params):
    """
    Check if a parameter combination has already been tested.

    Args:
        params: Parameter combination to check
        tested_params: List of previously tested parameter combinations

    Returns:
        Boolean indicating whether the combination has been tested
    """
    for tested in tested_params:
        match = True
        for key, value in params.items():
            if key == 'workers':
                continue

            if key == 'sample':
                if float(tested.get(key, 0)) != float(value):
                    match = False
                    break
            elif tested.get(key) != value:
                match = False
                break

        if match:
            return True

    return False


def grid_search_word2vec(corpus_dir, output_dir, param_grid, previous_results=None,
                         csv_pattern="*.csv", max_files=None, n_jobs=1):
    """
    Perform grid search to find optimal Word2Vec parameters.

    Args:
        corpus_dir: Directory containing corpus CSV files
        output_dir: Directory to save models and results
        param_grid: Dictionary of parameter names and possible values
        previous_results: Path to CSV file with previous results (None to start fresh)
        csv_pattern: Pattern to match corpus CSV files
        max_files: Maximum number of corpus files to load (None for all)
        n_jobs: Number of parallel jobs (currently not used)

    Returns:
        DataFrame containing all results sorted by accuracy
    """
    # Main hyperparameter optimization process: trains multiple Word2Vec models with different
    # settings, evaluates each on linguistic analogy tests, and tracks the best performer
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sentences = load_corpus_from_csv_files(corpus_dir, csv_pattern, max_files)

    if not sentences:
        print("No sentences loaded from corpus. Check your directory and file pattern.")
        return None

    tested_params = []
    if previous_results:
        tested_params = load_previously_tested_params(previous_results)

    param_combinations = list(dict(zip(param_grid.keys(), values))
                              for values in product(*param_grid.values()))

    new_param_combinations = []
    for params in param_combinations:
        if not is_already_tested(params, tested_params):
            new_param_combinations.append(params)

    print(
        f"Grid search will evaluate {len(new_param_combinations)} new parameter combinations")
    print(
        f"Skipping {len(param_combinations) - len(new_param_combinations)} already tested combinations")

    previous_results_df = pd.DataFrame()
    if previous_results and os.path.exists(previous_results):
        try:
            previous_results_df = pd.read_csv(previous_results)
            print(
                f"Loaded {len(previous_results_df)} previous results from {previous_results}")
        except Exception as e:
            print(f"Error loading previous results: {e}")

    results = []

    best_accuracy = 0
    if not previous_results_df.empty and 'overall_accuracy' in previous_results_df.columns:
        best_accuracy = previous_results_df['overall_accuracy'].max()
        print(f"Best accuracy from previous results: {best_accuracy:.4f}")

    for i, params in enumerate(new_param_combinations):
        print(f"\nParameter combination {i+1}/{len(new_param_combinations)}")
        print(params)

        result = train_and_evaluate(params, sentences, output_dir)
        results.append(result)

        accuracy = result.get('overall_accuracy', 0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model! Accuracy: {accuracy:.4f}")

        interim_df = pd.DataFrame(results)

        if not previous_results_df.empty:
            combined_df = pd.concat(
                [previous_results_df, interim_df], ignore_index=True)
        else:
            combined_df = interim_df

        combined_df = combined_df.sort_values(
            'overall_accuracy', ascending=False)

        interim_path = os.path.join(
            output_dir, f"word2vec_grid_search_interim_{timestamp}.csv")
        combined_df.to_csv(interim_path, index=False)

    if not previous_results_df.empty:
        results_df = pd.concat(
            [previous_results_df, pd.DataFrame(results)], ignore_index=True)
    else:
        results_df = pd.DataFrame(results)

    results_df = results_df.sort_values('overall_accuracy', ascending=False)

    results_path = os.path.join(
        output_dir, f"word2vec_grid_search_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if previous_results:
        results_df.to_csv(previous_results, index=False)
        print(f"Results also saved to {previous_results}")

    return results_df


if __name__ == "__main__":
    # Paths for corpus data, saved models, and parameter search results
    # Update these with your own directories relative paths
    corpus_dir = "final_07_09_postprocessed_data"
    output_dir = "new_models_07_09"
    # If continuing from a previous run, put the relative path to your csv file
    # If there is no interim csv file (First Run) set this as None
    previous_results = "new_models_07_09/word2vec_grid_search_interim_20250504_072335.csv"

    # Hyperparameter space for grid search
    # This is just the last iteration
    param_grid = {
        'vector_size': [300],
        'window': [5, 8, 10],
        'min_count': [5],
        'sg': [1],
        'hs': [0],
        'negative': [5, 15],
        'ns_exponent': [0.75],
        'alpha': [0.025],
        'min_alpha': [0.0001],
        'sample': [0.0001, 0.001],
        'epochs': [5],
        'workers': [21]
    }

    results = grid_search_word2vec(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        param_grid=param_grid,
        previous_results=previous_results,
        csv_pattern="enhanced_*.csv",
        max_files=None,
        n_jobs=1
    )

    if results is not None:
        print("\nBest parameter combination:")
        print(results.iloc[0])

"""
Analogy Test Script for Word2Vec models

This script evaluates how well a trained Word2Vec model can solve word analogies, which is a common
way to test the quality of word embeddings. The script uses pre-defined analogy test sets covering
various semantic and syntactic relationships.

Features:
- Evaluates model performance on different analogy categories
- Detailed metrics including accuracy, mean/median rank of expected words
- Export results to CSV for further analysis
- Select specific categories to test from the default sets
- Verbose output option for detailed per-analogy results

Usage:
    python word2vec_analogy_test.py --model model_path.model [--verbose] [--categories gender past-tense] [--output results.csv]

Some code in this module was adapted from code provided by Claude (Anthropic).
"""

import os
import argparse
import numpy as np
from gensim.models import Word2Vec
import csv
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DEFAULT_ANALOGIES = {
    "gender": [
        ("man", "woman", "king", "queen"),
        ("man", "woman", "husband", "wife"),
        ("man", "woman", "actor", "actress"),
        ("man", "woman", "father", "mother"),
        ("boy", "girl", "brother", "sister"),
        ("nephew", "niece", "uncle", "aunt"),
    ],
    "capital-country": [
        ("paris", "france", "rome", "italy"),
        ("berlin", "germany", "tokyo", "japan"),
        ("london", "england", "madrid", "spain"),
        ("washington", "usa", "moscow", "russia"),
        ("beijing", "china", "seoul", "korea"),
        ("ottawa", "canada", "canberra", "australia"),
    ],
    "comparative-superlative": [
        ("good", "best", "bad", "worst"),
        ("big", "biggest", "small", "smallest"),
        ("fast", "fastest", "slow", "slowest"),
        ("happy", "happiest", "sad", "saddest"),
        ("hard", "hardest", "easy", "easiest"),
        ("long", "longest", "short", "shortest"),
    ],
    "past-tense": [
        ("walk", "walked", "run", "ran"),
        ("see", "saw", "go", "went"),
        ("eat", "ate", "drink", "drank"),
        ("write", "wrote", "speak", "spoke"),
        ("play", "played", "grow", "grew"),
        ("fly", "flew", "fall", "fell"),
    ],
    "plural": [
        ("car", "cars", "child", "children"),
        ("dog", "dogs", "mouse", "mice"),
        ("house", "houses", "person", "people"),
        ("cat", "cats", "foot", "feet"),
        ("bird", "birds", "tooth", "teeth"),
        ("book", "books", "goose", "geese"),
    ],
    "opposite": [
        ("hot", "cold", "light", "dark"),
        ("up", "down", "front", "back"),
        ("open", "closed", "young", "old"),
        ("happy", "sad", "rich", "poor"),
        ("clean", "dirty", "fast", "slow"),
        ("east", "west", "north", "south"),
    ],
    "profession-place": [
        ("doctor", "hospital", "teacher", "school"),
        ("chef", "restaurant", "lawyer", "court"),
        ("farmer", "farm", "programmer", "office"),
        ("scientist", "laboratory", "pilot", "airplane"),
        ("librarian", "library", "actor", "theater"),
        ("baker", "bakery", "firefighter", "station"),
    ]
}


def load_model(model_path):
    """Load a trained Word2Vec model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = Word2Vec.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Vocabulary size: {len(model.wv.key_to_index):,} words")
        print(f"Vector dimensions: {model.wv.vector_size}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_analogy(model, word1, word2, word3, expected_word4, verbose=False):
    """
    Evaluate a single analogy: word1 is to word2 as word3 is to [expected_word4]
    Returns a tuple (success, prediction, rank, score)
    """
    # Skip if any word is not in vocabulary
    for word in [word1, word2, word3, expected_word4]:
        if word not in model.wv:
            if verbose:
                print(f"Word '{word}' not in vocabulary, skipping analogy")
            return {'success': False, 'prediction': None, 'rank': -1, 'score': 0.0, 'in_vocab': False}

    # Compute the analogy
    try:
        results = model.wv.most_similar(
            positive=[word2, word3], negative=[word1], topn=100)

        # Find the rank of the expected word
        prediction = results[0][0]
        prediction_score = results[0][1]

        # Check the rank of the expected word in results
        found = False
        rank = -1
        for i, (word, _) in enumerate(results):
            if word.lower() == expected_word4.lower():
                found = True
                rank = i + 1
                break

        success = prediction.lower() == expected_word4.lower()

        if verbose:
            result_str = "✓" if success else "✗"
            print(f"{result_str} {word1}:{word2} :: {word3}:{expected_word4}")
            print(f"  Predicted: {prediction} (score: {prediction_score:.4f})")
            if not success and found:
                print(f"  Expected '{expected_word4}' found at rank {rank}")
            if not success and not found:
                print(f"  Expected '{expected_word4}' not found in top 100")

        return {
            'success': success,
            'prediction': prediction,
            'rank': rank,
            'score': prediction_score,
            'in_vocab': True
        }

    except Exception as e:
        if verbose:
            print(f"Error evaluating analogy: {e}")
        return {'success': False, 'prediction': None, 'rank': -1, 'score': 0.0, 'in_vocab': True}


def run_analogy_tests(model, analogy_sets, verbose=False):
    """Run tests on multiple sets of analogies and report metrics."""
    overall_results = {
        'total': 0,
        'successful': 0,
        'in_vocab': 0,
        'all_in_vocab': 0,  # Count of analogies where all 4 words are in vocab
        'mean_rank': [],
        'median_rank': [],
        'by_category': {}
    }

    for category, analogies in analogy_sets.items():
        if verbose:
            print(f"\n=== Testing Category: {category} ===")

        category_results = {
            'total': len(analogies),
            'successful': 0,
            'in_vocab': 0,
            'mean_rank': [],
            'predictions': []
        }

        for analogy in analogies:
            word1, word2, word3, word4 = analogy

            result = evaluate_analogy(
                model, word1, word2, word3, word4, verbose)

            category_results['successful'] += int(result['success'])

            if result['in_vocab']:
                category_results['in_vocab'] += 1

                if result['rank'] > 0:
                    category_results['mean_rank'].append(result['rank'])
                    overall_results['mean_rank'].append(result['rank'])

                category_results['predictions'].append({
                    'analogy': analogy,
                    'predicted': result['prediction'],
                    'success': result['success'],
                    'rank': result['rank'],
                    'score': result['score']
                })

        # Calculate category metrics
        in_vocab_accuracy = category_results['successful'] / \
            category_results['in_vocab'] if category_results['in_vocab'] > 0 else 0
        total_accuracy = category_results['successful'] / \
            category_results['total'] if category_results['total'] > 0 else 0

        # Calculate mean rank for category
        if category_results['mean_rank']:
            category_results['mean_rank_value'] = np.mean(
                category_results['mean_rank'])
            category_results['median_rank_value'] = np.median(
                category_results['mean_rank'])
        else:
            category_results['mean_rank_value'] = 0
            category_results['median_rank_value'] = 0

        print(f"\nResults for category '{category}':")
        print(f"Total analogies: {category_results['total']}")
        print(f"Analogies with all words in vocabulary: {category_results['in_vocab']}")
        print(f"Successful predictions: {category_results['successful']}")
        print(f"Accuracy (vocab): {in_vocab_accuracy:.2%}")
        print(f"Accuracy (total): {total_accuracy:.2%}")

        if category_results['mean_rank']:
            print(
                f"Mean rank of expected word: {category_results['mean_rank_value']:.2f}")
            print(
                f"Median rank of expected word: {category_results['median_rank_value']:.2f}")

        overall_results['total'] += category_results['total']
        overall_results['successful'] += category_results['successful']
        overall_results['in_vocab'] += category_results['in_vocab']
        overall_results['by_category'][category] = category_results

    # Calculate overall metrics
    if overall_results['mean_rank']:
        overall_results['mean_rank_value'] = np.mean(
            overall_results['mean_rank'])
        overall_results['median_rank_value'] = np.median(
            overall_results['mean_rank'])
    else:
        overall_results['mean_rank_value'] = 0
        overall_results['median_rank_value'] = 0

    in_vocab_accuracy = overall_results['successful'] / \
        overall_results['in_vocab'] if overall_results['in_vocab'] > 0 else 0
    total_accuracy = overall_results['successful'] / \
        overall_results['total'] if overall_results['total'] > 0 else 0

    print("\n=== Overall Analogy Test Results ===")
    print(f"Total analogies tested: {overall_results['total']}")
    print(
        f"Analogies with all words in vocabulary: {overall_results['in_vocab']}")
    print(f"Successful predictions: {overall_results['successful']}")
    print(f"Accuracy (vocab): {in_vocab_accuracy:.2%}")
    print(f"Accuracy (total): {total_accuracy:.2%}")

    if overall_results['mean_rank']:
        print(
            f"Mean rank of expected word: {overall_results['mean_rank_value']:.2f}")
        print(
            f"Median rank of expected word: {overall_results['median_rank_value']:.2f}")

    return overall_results


def export_results(results, output_file):
    """Export test results to a CSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['Category', 'Word1', 'Word2', 'Word3',
                        'Expected', 'Predicted', 'Success', 'Rank', 'Score'])

        # Write data
        for category, category_results in results['by_category'].items():
            for prediction in category_results['predictions']:
                analogy = prediction['analogy']
                writer.writerow([
                    category,
                    analogy[0],
                    analogy[1],
                    analogy[2],
                    analogy[3],
                    prediction['predicted'],
                    int(prediction['success']),
                    prediction['rank'],
                    prediction['score']
                ])

    print(f"\nResults exported to {output_file}")





def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a Word2Vec model on analogy tasks')
    parser.add_argument('--model', '-m', required=True,
                        help='Path to the trained Word2Vec model')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed results for each analogy')
    parser.add_argument(
        '--output', '-o', default='analogy_results.csv', help='Output file for results')

    parser.add_argument('--categories', '-c', nargs='+',
                        help='Specific categories to test from the default set')

    args = parser.parse_args()

    model = load_model(args.model)
    if model is None:
        return



    test_sets = {}

    if args.categories:
        for category in args.categories:
            if category in DEFAULT_ANALOGIES:
                test_sets[category] = DEFAULT_ANALOGIES[category]
            else:
                print(
                    f"Warning: Category '{category}' not found in default analogies")
    else:
        test_sets = DEFAULT_ANALOGIES.copy()

    # Run tests
    results = run_analogy_tests(model, test_sets, args.verbose)

    # Export results
    if args.output:
        export_results(results, args.output)


if __name__ == "__main__":
    main()

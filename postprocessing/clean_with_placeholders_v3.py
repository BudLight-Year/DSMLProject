"""
Text Preprocessing Pipeline for Entity Recognition and Normalization

This script provides an efficient, parallel processing framework for cleaning and normalizing
text data with entity recognition. It focuses on processing CSV files containing tokens and
their associated entities, implementing intelligent normalization strategies to maintain
semantic meaning while reducing vocabulary size.

Key functionalities:
1. Entity frequency analysis to identify common entities worth preserving
2. Pattern-based entity classification (PERSON, ORG, LOC, etc.) for infrequent entities
3. Token normalization and filtering to remove noise
4. Multi-processing implementation for high-performance batch processing
5. Memory-efficient handling of large datasets through batched operations
6. Detailed statistics reporting for tracking processing effectiveness

Usage:
The script uses command-line arguments to control processing parameters including
input/output directories, frequency thresholds, and parallel processing options.

Some code in this module was adapted from code provided by Claude (Anthropic).
"""

import os
import glob
import re
import argparse
from tqdm import tqdm
import multiprocessing as mp
import time
import csv
from collections import Counter
import psutil

# Minimum frequency for entities to not be replaced with placeholders
ENTITY_FREQ_THRESHOLD = 5


def analyze_entity_frequencies(files, max_files=None, threshold=5):
    """
    Count entity frequencies and return ONLY the frequent entities (above threshold)
    to minimize memory usage and serialization overhead.
    """
    print("Analyzing entity frequencies...")

    # Use a sample of files for frequency analysis if specified
    # Mainly for testing
    if max_files is not None and max_files < len(files):
        sample_size = max_files
        sampled_files = files[:sample_size]
        print(f"Using {sample_size} files for entity frequency analysis")
    else:
        sampled_files = files
        print(f"Using all {len(files)} files for entity frequency analysis")

    entity_counter = Counter()
    entities_processed = 0

    for file_path in tqdm(sampled_files, desc="Counting entities"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    if len(row) > 1 and row[1]:  # Check if there are entities
                        entities = row[1].split('|')
                        for entity in entities:
                            if entity and entity.strip():
                                entity_counter[entity.strip()] += 1
                                entities_processed += 1
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")

    print(f"Processed {entities_processed:,} entity instances")
    print(f"Found {len(entity_counter):,} unique entities")

    # Filter non frequent entities
    frequent_entities = {entity: count for entity, count in entity_counter.items()
                         if count >= threshold}

    print(
        f"Entities appearing at least {threshold} times: {len(frequent_entities):,} ({len(frequent_entities)/len(entity_counter)*100:.1f}%)")
    print(
        f"Top 10 most common entities: {Counter(frequent_entities).most_common(10)}")

    # Report memory usage
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")

    return frequent_entities


def classify_entity(entity):
    """
    Determine the category of an entity using pattern matching.
    Very basic and can be significantly improved... given the time
    """
    # Check for sports teams (often have city names)
    if any(team in entity for team in ['Yankees', 'Lakers', 'United', 'Arsenal', 'Rangers', 'Rovers']):
        return "TEAM"

    # Check for person names (simple heuristic)
    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', entity) or "President" in entity:
        return "PERSON"

    # Check for organization names
    if any(org in entity for org in ['Inc', 'Corp', 'Company', 'Ltd', 'LLC', 'Association', 'Committee', 'Center']):
        return "ORG"

    # Check for locations
    if any(loc in entity for loc in ['City', 'County', 'State', 'York', 'Angeles', 'London', 'Paris']):
        return "LOC"

    # Numbers and quantities
    if re.match(r'^\d+$', entity) or re.match(r'^more than \d+$', entity) or re.match(r'^\d+\-\d+\-\d+$', entity):
        return "NUM"

    # Dates and times
    if any(month in entity.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        return "DATE"

    if re.match(r'^\d{4}$', entity):  # Years, or any other 4 digit numerical entity...
        return "YEAR"

    # Time expressions
    if any(time in entity.lower() for time in ['hour', 'minute', 'second', 'day', 'week', 'month', 'year']):
        return "TIME_EXPR"

    # Percentages
    if "percent" in entity.lower() or "%" in entity:
        return "PERCENT"

    # Money expressions
    if any(currency in entity for currency in ['$', '£', '€', 'dollar', 'euro', 'pound']):
        return "MONEY"

    # Default to MISC if we can't classify
    return "MISC"


def process_token(token):
    """
    Process tokens without using spaCy for better performance.
    """
    # Keep named entities (capitalized words)
    if re.match(r'^[A-Z][a-z]+$', token) and len(token) > 2:
        return token.lower()  # Lowercase for consistency

    # Replace numeric patterns with placeholders
    if re.match(r'^\d+(\.\d+)?$', token):
        return '<NUM>'

    # Replace time expressions
    if re.search(r'hour|minute|second|day|week|month|year', token, re.IGNORECASE):
        return '<TIME>'

    # Replace measurement expressions
    if re.search(r'meter|foot|inch|mile|kilometer|pound|kg|ton', token, re.IGNORECASE):
        return '<MEASURE>'

    # Filter out tokens with special characters
    if re.search(r'[/\\_\.\+\-]', token):
        return None

    # Filter out very short tokens (except common words)
    if len(token) < 3 and token.lower() not in ['a', 'an', 'the', 'is', 'am', 'be', 'to', 'in', 'on', 'at', 'by', 'it', 'of', 'or', 'and', 'for']:
        return None

    # Filter out tokens that look like noise or typos
    if len(token) >= 3:
        # No vowels likely means an abbreviation or noise
        if len(re.findall(r'[aeiou]', token.lower())) == 0:
            return None

        # Less than 70% alphabetic characters suggests noise
        if sum(1 for c in token if c.isalpha()) / len(token) < 0.7:
            return None

    # Keep all other tokens as they are
    return token.lower()  # Lowercase for consistency


def process_entity(entity, frequent_entities):
    """
    Process entities using the filtered frequent entities dictionary
    """
    if not entity or entity.lower() == 'nan':
        return None

    # Check if this is a frequent entity
    if entity in frequent_entities:
        # Normalize common patterns for frequent entities

        # Normalize number expressions
        if re.match(r'^\d+$', entity) or re.match(r'^more than \d+$', entity) or "percent" in entity.lower():
            return "<NUM>"

        # Normalize years
        if re.match(r'^\d{4}$', entity):  # Years
            return "<YEAR>"

        # Keep the entity as-is if it's frequent
        return entity
    else:
        # For infrequent entities, replace with category placeholder
        entity_type = classify_entity(entity)
        return f"<{entity_type}>"


def process_file(args):
    """Process a single file with batching for better memory usage"""
    file_path, output_path, frequent_entities, batch_size = args

    try:
        # Read the file with proper CSV handling
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
            rows = list(reader)

        # Process each row
        cleaned_rows = []
        total_tokens_before = 0
        total_tokens_after = 0

        for row in rows:
            if not row:
                continue

            if len(row) > 0:
                tokens_text = row[0]
                tokens = tokens_text.split()
                total_tokens_before += len(tokens)

                clean_tokens = []
                for token in tokens:
                    processed = process_token(token)
                    if processed:
                        clean_tokens.append(processed)

                total_tokens_after += len(clean_tokens)

                entities_text = row[1] if len(row) > 1 else ""

                # Process entities
                if entities_text and entities_text.lower() != "nan":
                    entities = entities_text.split('|')
                    clean_entities = []

                    for entity in entities:
                        processed_entity = process_entity(
                            entity, frequent_entities)
                        if processed_entity:
                            clean_entities.append(processed_entity)

                    clean_entities_text = '|'.join(
                        clean_entities) if clean_entities else ""
                else:
                    clean_entities_text = ""

                # Only add if there are tokens left
                if clean_tokens:
                    cleaned_row = [' '.join(clean_tokens), clean_entities_text]
                    cleaned_rows.append(cleaned_row)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(cleaned_rows)

        return {
            'tokens_before': total_tokens_before,
            'tokens_after': total_tokens_after,
            'rows_before': len(rows),
            'rows_after': len(cleaned_rows)
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {
            'tokens_before': 0,
            'tokens_after': 0,
            'rows_before': 0,
            'rows_after': 0
        }


def clean_processed_files(input_dir, output_dir, file_pattern='*.csv', num_processes=None,
                          max_files=None, entity_threshold=5, batch_size=1000):
    """
    Enhanced cleaning of processed token files with efficient entity handling

    Focuses on parallel processing the data
    """
    print(f"Current working directory: {os.getcwd()}")

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    print(f"Absolute input directory: {input_dir}")
    print(f"Absolute output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    print(f"Using {num_processes} processes for parallel processing")

    # Get list of files
    pattern = os.path.join(input_dir, file_pattern)
    print(f"Looking for files with pattern: {pattern}")

    all_files = sorted(glob.glob(pattern))

    # Limit number of files for testing
    if max_files is not None:
        all_files = all_files[:max_files]
        print(f"Limited to processing {max_files} files")

    print(f"Found {len(all_files)} files matching pattern: {pattern}")

    if not all_files:
        print(
            "No files found. Check that the input directory and file pattern are correct.")
        print("Directory contents:")
        try:
            print(os.listdir(input_dir))
        except Exception as e:
            print(f"Error listing directory: {e}")
        return

    global ENTITY_FREQ_THRESHOLD
    ENTITY_FREQ_THRESHOLD = entity_threshold

    # First pass - analyze entity frequencies and filter to keep only frequent entities
    frequent_entities = analyze_entity_frequencies(
        all_files, max_files=None, threshold=entity_threshold)

    # Prepare arguments for parallel processing
    args_list = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"enhanced_{file_name}")
        args_list.append(
            (file_path, output_path, frequent_entities, batch_size))

    # Process files in parallel
    start_time = time.time()

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, args_list),
            total=len(args_list),
            desc="Cleaning files"
        ))

    # Summarize results
    total_tokens_before = sum(stats['tokens_before'] for stats in results)
    total_tokens_after = sum(stats['tokens_after'] for stats in results)
    total_rows_before = sum(stats['rows_before'] for stats in results)
    total_rows_after = sum(stats['rows_after'] for stats in results)

    elapsed_time = time.time() - start_time

    print("\nProcessing complete!")
    print(
        f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    if all_files:
        print(
            f"Processing rate: {len(all_files)/elapsed_time:.2f} files/second")

        if total_tokens_before > 0:
            token_percentage = total_tokens_after/total_tokens_before*100
            print(
                f"Total tokens: {total_tokens_before:,} → {total_tokens_after:,} ({token_percentage:.1f}%)")
        else:
            print(
                f"Total tokens: {total_tokens_before:,} → {total_tokens_after:,}")

        if total_rows_before > 0:
            row_percentage = total_rows_after/total_rows_before*100
            print(
                f"Total rows: {total_rows_before:,} → {total_rows_after:,} ({row_percentage:.1f}%)")
        else:
            print(f"Total rows: {total_rows_before:,} → {total_rows_after:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Efficient cleaning of tokens with better entity handling')
    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing processed files')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Directory to save cleaned files')
    parser.add_argument('--file-pattern', '-p',
                        default='*.csv', help='Pattern to match files')
    parser.add_argument('--num-processes', '-n', type=int,
                        default=None, help='Number of processes to use')
    parser.add_argument('--max-files', '-m', type=int,
                        default=None, help='Maximum number of files to process')
    parser.add_argument('--entity-threshold', '-e', type=int, default=5,
                        help='Frequency threshold for entities (default: 5)')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                        help='Number of rows to process in each batch (default: 1000)')

    args = parser.parse_args()

    clean_processed_files(
        args.input_dir,
        args.output_dir,
        args.file_pattern,
        args.num_processes,
        args.max_files,
        args.entity_threshold,
        args.batch_size
    )

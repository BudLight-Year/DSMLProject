"""
Functions and script to preprocess text data for an
embedding model to extract semantic relationships.

Utilizes spaCy to preprocess code including
    - Tokenization
    - Named entity recognition
    - Vectorization
    - Stop-word removal
    - Part of speech tagging
    - Lemmatization

Uses parallel and batch processing to accelerate the pre processing.
Saves pre processed data in files.

Some code in this module was adapted from code provided by Claude (Anthropic).
Mainly used to help switch from nltk to spaCy,
utilizing efficient batch and parallel processing,
and creating cli using Argument Parser.
"""


import os
import csv
import time
import psutil
import multiprocessing as mp
from tqdm import tqdm
import spacy
import re
import argparse


# -------------------------------- #
#                                  #
#          spaCy functions         #
#                                  #
# -------------------------------- #

def handle_social_media_elements(text: str) -> str:
    """Replace URLs, usernames, and hashtags with placeholders."""
    text = re.sub(r'https?://\S+', '<URL>', text)
    text = re.sub(r'www\.\S+\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'#(\w+)', r'\1 <HASHTAG>', text)

    return text


def preserve_entity_patterns(text: str) -> str:
    """Preserve patterns like dates, times, and currency."""
    date_pattern = r"\b(?:[A-Z][a-z]*\s\d{1,2},\s\d{4}|\d{1,2}/\d{1,2}/\d{4})\b"
    time_pattern = r"\b\d{1,2}:\d{2}\s?[apm\.]+\b"
    currency_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?"

    text = re.sub(date_pattern, lambda m: f"__DATE_{m.group(0).replace(' ', '_').replace(',', '').replace('.', '')}__", text)
    text = re.sub(time_pattern, lambda m: f"__TIME_{m.group(0).replace(':', '').replace(' ', '_').replace('.', '')}__", text)
    text = re.sub(currency_pattern, lambda m: f"__CURRENCY_{m.group(0).replace('$', '')}__", text)

    return text


def init_spacy_nlp():
    """
    Initialize and configure spaCy NLP pipeline.

    Loads the NLP model disabling unused components.
    Creates a custom tokenizer which calls handle_social_media_elements and
    preserve_entity_pattern before the default tokenizer.
    Assigns the custom tokenizer to the model.
    """
    nlp = spacy.load("en_core_web_lg", disable=["parser", "textcat"])
    # Increase the amount of data that can be processed at once
    nlp.max_length = 2000000

    default_tokenizer = nlp.tokenizer

    def custom_tokenizer(text):
        text = handle_social_media_elements(text)
        text = preserve_entity_patterns(text)
        return default_tokenizer(text)

    nlp.tokenizer = custom_tokenizer

    return nlp


def preprocess_with_spacy(text, nlp):
    """
    Process text using spaCy for:
    - Named entity recognition
    - Tokenization
    - Stop-word removal
    - Lemmatization
    """
    # Tokenize, NER, and vectorize
    doc = nlp(text)

    # Extract named entities
    entities = [ent.text for ent in doc.ents]

    # Lemmatization loop
    processed_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        # Retrieve a lemmatized, lowercased version of token
        lemma = token.lemma_.lower()

        # Checks for entity patterns
        # If pattern found, reformat text and skip lemmatization
        if token.text.startswith('__'):
            if '__DATE_' in token.text or '__TIME_' in token.text or '__CURRENCY_' in token.text:
                marker_parts = token.text.strip('__').split('_', 1)
                if len(marker_parts) > 1:
                    marker_content = marker_parts[1].lower()
                    subtokens = re.findall(r'\w+', marker_content)
                    processed_tokens.extend(subtokens)
                continue

        # Check if empty
        if lemma and lemma.strip():
            processed_tokens.append(lemma)

    return processed_tokens, entities


# -------------------------------- #
#                                  #
#     Multiprocessing and I/O      #
#                                  #
# -------------------------------- #

def initialize_worker():
    """Initialize the spaCy NLP pipeline for each worker process."""
    global nlp
    nlp = init_spacy_nlp()


def process_chunk(chunk_lines):
    """Process a chunk of lines using spaCy-only preprocessing."""
    processed_records = []

    for line in chunk_lines:
        line = line.strip()
        if not line:
            continue

        try:
            tokens, entities = preprocess_with_spacy(line, nlp)

            record = {
                "tokens": tokens,
                "entities": entities
            }

            processed_records.append(record)

        except Exception as e:
            print(f"Error processing line: {e}")
            continue

    return processed_records


def read_in_chunks(file_obj, chunk_size=10*1024*1024):
    """Generator to read a file in chunks efficiently."""
    pending = ''
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break

        text = pending + chunk

        lines = text.split('\n')
        if chunk:
            pending = lines[-1]
            lines = lines[:-1]
        else:
            pending = ''

        for line in lines:
            if line:
                yield line.strip()

    if pending:
        yield pending.strip()


def write_records_efficient(records, output_path, format_type='csv', append=False):
    """Write records to the output file using CSV module."""
    mode = 'a' if append else 'w'

    with open(output_path, mode, encoding='utf-8', newline='') as f:
        if format_type == 'jsonl':
            import json
            for record in records:
                json_line = json.dumps(record)
                f.write(json_line + '\n')

        elif format_type == 'tokens_only':
            for record in records:
                tokens_line = ' '.join(record['tokens'])
                f.write(tokens_line + '\n')

        elif format_type == 'csv':
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            for record in records:
                tokens_str = ' '.join(record['tokens'])
                entities_str = '|'.join(record['entities']) if record['entities'] else ''
                writer.writerow([tokens_str, entities_str])

    return os.path.getsize(output_path)


def count_lines_efficiently(file_path):
    """Count lines in a file efficiently."""
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def process_file_optimized(input_file_path, output_dir, base_output_name=None, 
                           max_file_size_mb=45, num_processes=None, 
                           processing_batch_size=10000, write_batch_size=50000,
                           show_progress=True, save_format='csv'):
    """
    Process a large text file using optimized I/O and multiprocessing.

    Args:
        input_file_path (str): Path to the input file
        output_dir (str): Directory to save output files
        base_output_name (str, optional): Base name for output files. If None, uses the input filename
        max_file_size_mb (int): Maximum size of each output file in MB
        num_processes (int): Number of processes to use (default: CPU count - 1)
        processing_batch_size (int): Number of lines per processing batch
        write_batch_size (int): Number of records to accumulate before writing to disk
        show_progress (bool): Whether to show progress bar
        save_format (str): Format to save data ('jsonl', 'tokens_only', or 'csv')
    """
    os.makedirs(output_dir, exist_ok=True)

    if base_output_name is None:
        input_filename = os.path.basename(input_file_path)
        base_name = f"processed_{os.path.splitext(input_filename)[0]}"
    else:
        base_name = base_output_name

    print(f"Using base name for output files: {base_name}")

    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    print(f"Using {num_processes} processes for parallel processing")

    if show_progress:
        print("Counting lines in the file...")
        total_lines = count_lines_efficiently(input_file_path)
        print(f"Total lines to process: {total_lines:,}")
    else:
        total_lines = None

    start_time = time.time()
    file_counter = 1
    current_output_path = os.path.join(output_dir, f"{base_name}.{file_counter:04d}.{save_format}")
    all_processed_records = []

    with mp.Pool(processes=num_processes, initializer=initialize_worker) as pool:
        pbar = tqdm(total=total_lines) if show_progress and total_lines else None

        lines_processed = 0
        pending_lines = []

        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in read_in_chunks(f):
                pending_lines.append(line)

                if len(pending_lines) >= processing_batch_size:
                    chunks_for_processes = []
                    chunk_size = max(1, len(pending_lines) // num_processes)

                    for i in range(0, len(pending_lines), chunk_size):
                        chunks_for_processes.append(pending_lines[i:i+chunk_size])

                    results = pool.map(process_chunk, chunks_for_processes)

                    batch_results = [record for sublist in results for record in sublist]
                    all_processed_records.extend(batch_results)

                    lines_processed += len(pending_lines)
                    if pbar:
                        pbar.update(len(pending_lines))

                    pending_lines = []

                    if len(all_processed_records) >= write_batch_size:
                        estimated_record_size = len(str(all_processed_records[0])) * len(all_processed_records)

                        if os.path.exists(current_output_path):
                            current_size = os.path.getsize(current_output_path)
                            if current_size + estimated_record_size > max_file_size_mb * 1024 * 1024:
                                file_counter += 1
                                current_output_path = os.path.join(output_dir, f"{base_name}.{file_counter:04d}.{save_format}")
                                append = False
                            else:
                                append = True
                        else:
                            append = False

                        write_records_efficient(
                            all_processed_records, 
                            current_output_path, 
                            format_type=save_format,
                            append=append
                        )

                        all_processed_records = []

                    if show_progress and lines_processed % 100000 < processing_batch_size:
                        elapsed = time.time() - start_time
                        rate = lines_processed / elapsed
                        if total_lines:
                            eta_seconds = (total_lines - lines_processed) / rate if rate > 0 else 0
                            eta_hours = eta_seconds // 3600
                            eta_minutes = (eta_seconds % 3600) // 60
                            progress_info = f"{lines_processed:,}/{total_lines:,} lines ({lines_processed/total_lines*100:.1f}%)"
                            eta_info = f"ETA: {int(eta_hours)}h {int(eta_minutes)}m"
                        else:
                            progress_info = f"{lines_processed:,} lines"
                            eta_info = ""

                        memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

                        print(f"\nProcessed {progress_info}. "
                              f"Rate: {rate:.2f} lines/sec. "
                              f"{eta_info} | "
                              f"Memory: {memory_usage:.1f} MB | "
                              f"Current file: {file_counter}")

            if pending_lines:
                chunks_for_processes = []
                chunk_size = max(1, len(pending_lines) // num_processes)

                for i in range(0, len(pending_lines), chunk_size):
                    chunks_for_processes.append(pending_lines[i:i+chunk_size])

                results = pool.map(process_chunk, chunks_for_processes)

                batch_results = [record for sublist in results for record in sublist]
                all_processed_records.extend(batch_results)

                lines_processed += len(pending_lines)
                if pbar:
                    pbar.update(len(pending_lines))

    if all_processed_records:
        if os.path.exists(current_output_path):
            current_size = os.path.getsize(current_output_path)
            if current_size + (len(str(all_processed_records[0])) * len(all_processed_records)) > max_file_size_mb * 1024 * 1024:
                file_counter += 1
                current_output_path = os.path.join(output_dir, f"{base_name}.{file_counter:04d}.{save_format}")
                append = False
            else:
                append = True
        else:
            append = False

        write_records_efficient(
            all_processed_records,
            current_output_path,
            format_type=save_format,
            append=append
        )

    if pbar:
        pbar.close()

    elapsed_time = time.time() - start_time
    print("\nProcessing complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Processing rate: {lines_processed/elapsed_time:.2f} lines/second")
    print(f"Total output files: {file_counter}")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description='Process a large text file with optimized SpaCy-only NLP preprocessing')
    parser.add_argument('--input', '-i', required=True, help='Path to the input file')
    parser.add_argument('--output-dir', '-o', default='processed_data', help='Directory to save output files')
    parser.add_argument('--base-name', '-b', default=None, 
                        help='Base name for output files (default: derived from input filename)')
    parser.add_argument('--max-size', '-m', type=int, default=45, help='Maximum file size in MB')
    parser.add_argument('--processes', '-p', type=int, help='Number of processes (default: CPU count - 1)')
    parser.add_argument('--proc-batch', type=int, default=10000, help='Processing batch size')
    parser.add_argument('--write-batch', type=int, default=50000, help='Write batch size')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--format', '-f', choices=['jsonl', 'csv', 'tokens_only'], default='csv',
                        help='Output format (default: csv)')

    args = parser.parse_args()

    process_file_optimized(
        input_file_path=args.input,
        output_dir=args.output_dir,
        base_output_name=args.base_name,
        max_file_size_mb=args.max_size,
        num_processes=args.processes,
        processing_batch_size=args.proc_batch,
        write_batch_size=args.write_batch,
        show_progress=not args.no_progress,
        save_format=args.format
    )

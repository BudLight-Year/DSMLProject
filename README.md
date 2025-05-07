Overview:
    This projects goal is to compare semantic embeddings for english text from 2007-2009 to 2024 to discover semantic changes over 15 years. This will be accomplished by pre processing english text data from the two periods and training a Word2Vec model for each period. Then the results of the embedded words will be compared against each other to highilight any interesting changes.

Data:
    Uses data from https://data.statmnt.org/news-crawl/ 
    Data is snippets from online articles in the english language from 2007-2024.
    Data was only used from the 2007, 2008, 2009, and 2024 datasets.

Models:
    The model utilized for the semantic embeddings is a Word2Vec model from gensim. A large model pre trained by spaCy was used for pre processing.

Tools:
    spaCy (For NLP pre processing steps)
    Gensim Word2Vec model (For creating semantic embeddings)
    Pandas (for loading data into the model)

How to Run:
    Assuming the datasets are in their own directories, we call optimized_processor_spacy.py using the data directory as input. Using argparse you can feed parameters in using the cli. After getting processed data we call training_script.py using the processed data as input.

    optimized_processor_spacy:
        call giving arguments for input and ouput paths, output file name, maximum size of output chunked files, how many processes to run simultaneously, batch size for processing and writing, and format of the output files

        This script will preprocess text data from the format of the text data in the datasets (columns: )
        It uses spaCy pipeline without the parser or textcat (since we wont be using these).
        It leverages batch and parallel processing, and saves processed data in chunked files in a size set by user on run.

    training_script.py:
        call giving arguments for input directory, output directory, file pattern for extension type, vector size, context window size, minimum word frequency to train on a word, number of cpu cores to use, skipgram or CBOW, number of epochs, sample size (for testing), years (for file filtering).

        This script will train a gensim Word2Vec model using either skipgram or CBOW (depending on user input).
        It loads the chunked files and trains the model using user input parameters. Also leverages parallel and batch processing.
        For my training I used:
        vectors:                300
        context window size:    5
        min count:              5
        type:                   CBOW
        epochs:                 5

    

Contains files for 
    preprocessing data
    training a Word to Vector model


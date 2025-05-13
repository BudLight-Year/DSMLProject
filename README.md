# Semantic Change Analysis (2007-2009 vs 2024)

## Overview
This project aims to compare semantic embeddings for English text from 2007-2009 to 2024 to discover semantic changes over 15 years. This is accomplished by preprocessing English text data from the two periods and training a Word2Vec model for each period. The results of the embedded words are then compared against each other to highlight interesting semantic changes.

## Data
- Uses data from https://data.statmnt.org/news-crawl/
- Data consists of snippets from online articles in the English language
- Only data from 2007, 2008, 2009, and 2024 datasets was used

## Models
The model utilized for the semantic embeddings is a Word2Vec model from Gensim. A large model pre-trained by spaCy was used for preprocessing.

You can download the final versions of the models here:
- [2007-2009 Word2Vec Model](https://huggingface.co/Celeryman7/07_09_word2vec/tree/main)
- [2024 Word2Vec Model](https://huggingface.co/Celeryman7/24word2vec/tree/main)

## Tools
- **spaCy**: For NLP preprocessing steps
- **Gensim Word2Vec**: For creating semantic embeddings
- **Pandas**: For loading data into the model

## How to Run

> **NOTE**: These scripts use argparse for flexibility. If you are unsure what arguments you need to pass or the signatures used to pass them, you can run `file_name.py --help` in the console to see all argument options.

> **NOTE**: This process requires two different virtual environments to avoid dependency conflicts.

> **NOTE**: If you want to download the pre-trained models to test the semantic analysis, skip ahead to the model analysis section.

Since the dataset is too large to hold in this repo, it will have to be downloaded and added manually from the data source stated above in the Data section. Make sure to use the shuffled versions of the data.

I used data from 2007-2009 and 2024, but any time periods can be chosen. Just ensure that the data for period1 and the data for period2 are in different directories.

### Preprocessing

Before processing the data, build and activate a venv using the `processing_requirements.txt`.

#### Run optimized_processor_spacy.py:
Call giving required arguments for input and output paths.

Example call:
```bash
python optimized_processer_spacy.py --input ../Data/Period1 --output-dir ../PreProcessedData/Period1
```

This script will preprocess text data from the format of the text data in the datasets. It uses spaCy pipeline without the parser or textcat (since we won't be using these). It leverages batch and parallel processing, and saves processed data in chunked files (default size is 45MB per file).

Preprocessing steps:
- NER (Named Entity Recognition)
- Stop-word removal
- Tokenization
- Lemmatization

### Postprocessing

#### Run clean_with_placeholders_v3.py:
Call giving required arguments for input directory and output directory.

Example call:
```bash
python clean_with_placeholders_v3.py --input-dir ../PreProcessedData/Period1 --output-dir ../PostProcessedData/Period1
```

This script will take preprocessed data from the optimized_processor_spacy.py and further refine it. Its main job is to prune noise in the data, specifically within entities. It identifies entities that appear infrequently and replaces them with placeholders such as `<PEOPLE>`. This removes the noise from the data while still maintaining semantic context in the text.

### Training

Before running these scripts, build and activate a venv using `model_requirements.txt`

#### Run grid_search.py (optional):
This script is optional and not required to replicate the results of the pre-trained models (Since the values derived from this grid_search are used as defaults in training_script.py).

If you do want to use this file, make sure to change the three arguments at the start of the `if __name__ == "__main__":` block to match your directory structure and file names.

This script will take a word2vec model and conduct a comprehensive grid search using the parameters in the param_grid. It is not very efficient and requires user intervention to prevent long executions for minimal insights. After training a model, the script will conduct a quick accuracy test using some analogies and will log the results in a CSV file. The script can load previous runs from a CSV file so it can run results for one variation of the param_grid, stop, then resume without executing any already tested parameter combos.

#### Run training_script.py:
Call passing required arguments for input directory (Directory of the processed data). It is recommended to always declare an output name as well (name the model gets saved to).

Example Call:
```bash
python training_script.py --input-dir PostProcessedData --output word2vec_period1.model
```

This script will train a Gensim Word2Vec model using either skipgram or CBOW (skip-gram by default).
It loads the chunked files and trains the model using user input parameters. Also leverages parallel and batch processing.

Default parameters:
- Vector Size: 300
- Context Window: 5 words
- Minimum Word Count: 5 (how often words must appear to be used)
- Negative Sampling: 5 negative samples
- Learning Rate: 0.025 (decays to 0.0001)
- Epochs: 5
- sg: 1 (skip-gram)

The argparser has these values as defaults so no need to explicitly pass these arguments, but feel free to change any to fit your needs.

### Model Analysis

> **NOTE**: These scripts also require the model venv to be active.

#### Run analogies.py:
Call passing required argument for `--model`. It is recommended to pass an argument for `--output` as well.

Example Call:
```bash
python analogies.py --model ../Period1Model/word2vec_period1.model --output period1_analogy_results.csv
```

This script evaluates models based on their performance on a series of hard-coded analogies. It tracks metrics like accuracy and saves results in a CSV file.

#### Run general_analysis.py:
Call passing required arguments for `--models` in format: period:path. It is recommended to provide an argument for `--output-dir` as well.

Example Call:
```bash
python general_analysis.py --models period1:../Period1Model/word2vec_period1.model period2:../Period2Model/word2vec_period2.model --output-dir comparison_results
```

This script does comprehensive analysis on two different models. It:
- Identifies most common words and most unique words between models
- Compares semantic shift of words between models
- Visualizes semantic space using dimensionality reduction
- Identifies words with largest semantic shifts across models
- Compares word:context relationships across models

### Demo Scripts
These are scripts with no argparse designed to be run with minimal setup to demonstrate a specific type of semantic analysis.

These scripts require the model venv to be active.

#### interactive_analogies.py:
Ensure you have two trained models in the project directory.
Assign the path and names of the models to the variables at the top of the file.

This script will provide a simple command line interface where you can insert three words, letting the model predict the fourth word.
Example: "man woman king" would test "man is to woman as king is to ___" (Where the model should predict queen)

#### visualization_demo.py
Ensure you have two trained models in the project directory.
Assign the paths and names of your models in the main portion of the script at the bottom.
Insert any words you want to see on the visualization inside the `demo_words` list.

This script will create a 2D semantic space visual using dimensionality reduction for each model.
You can then manually place the visuals side by side to inspect differences and similarities.

## Results

Before implementing the post-processing step, the models had minimal insights due to the volume of noise in the data. After post-processing, the models provide good insight into the semantic changes in words across periods. Despite this, viewing the results of any analysis requires diligence as some results may look like genuine insights but could be products of the model/data and not the language.

According to the analysis results, it appears that most words have undergone significant semantic shift across the two periods. It is debatable if all of these words have really changed that much, or if the data from previous years does not have a proper semantic representation.

The models perform well on common words and somewhat rare words, but they do poorly on rare words. Regardless, they perform well enough to warrant analysis of the semantic results. The results show interesting changes in relationships between words, especially in the context of social media platforms and how they form stronger semantic relationships with internal terminology than they do with external terms.

For example, YouTube in 2007-2009 was most closely related to external terminology, namely other social media platforms, but in 2024, YouTube most closely relates to internal terms like usernames. Interactive analysis like the demo for interactive analogies is enjoyable to use, even for a layman. Sometimes you receive outputs that show interesting relationships and sometimes you get funny results such as "whale is to ocean as giraffe is to lava".

### Limitations of the Data

- The data is sourced only from online news articles. These articles have a particular way of representing written text that does not reflect speech in other forums. This means that the data does not have valuable semantic information you could get from text from sources like Twitter, Reddit, Facebook, etc.
- Data is fairly limited in early years and may not be as good quality as data from more recent years. Before removing noise, there was more representation of typos in the 2007-2009 data compared to the 2024 data.

### Limitations of the Model

Word2Vec is a great model due to its performance and its easy usability/quick training time. It does have several downsides when used for this kind of semantic analysis across time periods:

- It creates static embeddings for words, meaning it does not take into account words that have multiple meanings.
- It doesn't capture grammatical relationships since it focuses on "distributional semantics".
- It does not perform well with morphological context. This is because different forms of a word will be treated as completely separate words.

## Next Steps

The first next step would be to address the limitations of the data and the model:

- Sourcing additional data would be beneficial, yet a challenge for early years. Being able to use scanned physical publications would also be helpful, especially if I decide to analyze a time period before or at the beginning of the internet.
- Experimenting with additional models would also be a good next step. Though the bulk of work for this project was to refine the data specifically for a Word2Vec model, so experimenting with different models would require completely different preprocessing steps. For example, some models can actually utilize stop words to retrieve additional semantic relationships, specifically in a grammatical context. I am interested in the BERT model as it seems to be able to capture more in-depth semantic relationships.

After addressing these limitations, I would explore the practical use of this analysis and see if there is some useful application that could be created with it. An easy application is to utilize the data as input for a more sophisticated deep learning model.
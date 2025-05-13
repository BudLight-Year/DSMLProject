"""
Interactive analogy testing for two Word2Vec models

Made to demo analogies in front of a live audience

Run script and input three words delimited by spaces.

See funky business.
"""

from gensim.models import Word2Vec

MODEL1_PATH = "final_07_09_model/final_07_09.model"
MODEL2_PATH = "final_24_model/final_2024_skipgram.model"
MODEL1_NAME = "'07 '09 Model"
MODEL2_NAME = "'24 Model"


def load_model(model_path):
    """Load a trained Word2Vec model."""
    try:
        model = Word2Vec.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Vocabulary size: {len(model.wv.key_to_index)} words")
        print(f"Vector dimensions: {model.wv.vector_size}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def process_analogy(model1, model2, word1, word2, word3):
    """Process an analogy across two models and display results."""
    print(f"\nAnalogy: {word1} is to {word2} as {word3} is to ?")

    for model, name in [(model1, MODEL1_NAME), (model2, MODEL2_NAME)]:
        missing_words = []
        for word in [word1, word2, word3]:
            if word not in model.wv:
                missing_words.append(word)

        if missing_words:
            print(
                f"\n[{name}] Words not in vocabulary: {', '.join(missing_words)}")
            continue

        # Compute the analogy
        try:
            results = model.wv.most_similar(
                positive=[word2, word3], negative=[word1], topn=5)

            print(f"\nResults for {name}:")
            for i, (word, score) in enumerate(results):
                print(f"{i+1}. {word} (similarity: {score:.4f})")

        except Exception as e:
            print(f"\n[{name}] Error: {e}")


def interactive_mode(model1, model2):
    """Interactive mode for testing analogies against two models."""
    print("\n=== Interactive Analogy Testing ===")
    print("Enter analogies in the format: word1 word2 word3")
    print("Example: man woman king (to find queen)")
    print("Type 'exit' to quit")

    while True:
        try:
            user_input = input("\nEnter analogy > ").strip()

            if user_input.lower() in ('exit', 'quit', 'q'):
                break

            parts = user_input.split()
            if len(parts) != 3:
                print("Please enter exactly 3 words in the format: word1 word2 word3")
                continue

            word1, word2, word3 = parts
            process_analogy(model1, model2, word1, word2, word3)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Exiting interactive mode")


def main():
    """Main program entry point."""
    print("Loading models...")

    model1 = load_model(MODEL1_PATH)
    model2 = load_model(MODEL2_PATH)

    if model1 is None or model2 is None:
        print("Failed to load one or both models. Exiting.")
        return

    interactive_mode(model1, model2)


if __name__ == "__main__":
    main()

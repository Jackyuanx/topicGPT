import pandas as pd
import traceback
from topicgpt_python.utils import APIClient
import argparse
from sentence_transformers import SentenceTransformer, util
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sbert = SentenceTransformer("all-MiniLM-L6-v2")


def generate_comparison(api, model, data_file, prompt_file, output_file, verbose):
    api_client = APIClient(api=api, model=model)

    # Load documents
    df = pd.read_json(data_file, lines=True)

    if len(df) < 2:
        raise ValueError("Input file must contain exactly two documents for comparison.")

    doc1 = df.iloc[0]['prompted_docs']
    doc2 = df.iloc[1]['prompted_docs']

    # Read prompt
    with open(prompt_file, 'r') as f:
        comparison_prompt_template = f.read()

    combined_docs = f"Document 1:\n{doc1}\n\nDocument 2:\n{doc2}"

    prompt = comparison_prompt_template.format(document=combined_docs)

    try:
        response = api_client.iterative_prompt(
            prompt,
            max_tokens=2000,
            temperature=0.0,
            top_p=1.0,
            verbose=verbose
        )

        if verbose:
            print("Comparison Response:", response)

    except Exception as e:
        traceback.print_exc()
        response = "Error generating comparison"

    # Write results
    with open(output_file, 'w') as f:
        f.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--api", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()

    generate_comparison(
        api=args.api,
        model=args.model,
        data_file=args.data_file,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        verbose=args.verbose
    )
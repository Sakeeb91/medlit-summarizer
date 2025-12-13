"""
Data preparation module for Medical Literature Summarizer.
Loads and processes PubMed summarization dataset from HuggingFace.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def load_pubmed_dataset(split: str = "train", num_samples: int = 1000):
    """
    Load PubMed summarization dataset from HuggingFace.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        num_samples: Number of samples to load (None for all)

    Returns:
        List of dictionaries with 'article' and 'abstract' keys
    """
    print(f"Loading PubMed summarization dataset ({split} split)...")
    dataset = load_dataset("ccdv/pubmed-summarization", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    data = []
    for item in tqdm(dataset, desc="Processing samples"):
        data.append({
            "article": item["article"],
            "abstract": item["abstract"]
        })

    print(f"Loaded {len(data)} samples")
    return data


def create_training_conversations(data: list) -> list:
    """
    Convert article/abstract pairs into conversation format for fine-tuning.

    Args:
        data: List of dictionaries with 'article' and 'abstract' keys

    Returns:
        List of conversation dictionaries
    """
    conversations = []

    system_prompt = """You are an expert medical literature summarizer. Your task is to read medical research articles and produce clear, accurate, and concise summaries that capture the key findings, methodology, and conclusions. Focus on:
- Main research objectives and hypotheses
- Key methodology and study design
- Important results and statistical findings
- Clinical implications and conclusions

Maintain scientific accuracy while making the summary accessible."""

    for item in tqdm(data, desc="Creating conversations"):
        # Truncate very long articles to fit context window
        article = item["article"][:12000]  # ~4000 tokens approx

        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following medical research article:\n\n{article}"
                },
                {
                    "role": "assistant",
                    "content": item["abstract"]
                }
            ]
        }
        conversations.append(conversation)

    return conversations


def save_conversations(conversations: list, output_path: str):
    """Save conversations to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Saved {len(conversations)} conversations to {output_path}")


def prepare_data(
    train_samples: int = 800,
    val_samples: int = 100,
    test_samples: int = 100,
    output_dir: str = "data"
):
    """
    Prepare train, validation, and test datasets.

    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        output_dir: Output directory for processed data
    """
    output_dir = Path(output_dir)

    # Load datasets
    train_data = load_pubmed_dataset("train", train_samples)
    val_data = load_pubmed_dataset("validation", val_samples)
    test_data = load_pubmed_dataset("test", test_samples)

    # Create conversations
    train_conversations = create_training_conversations(train_data)
    val_conversations = create_training_conversations(val_data)
    test_conversations = create_training_conversations(test_data)

    # Save to JSONL
    save_conversations(train_conversations, output_dir / "train.jsonl")
    save_conversations(val_conversations, output_dir / "val.jsonl")
    save_conversations(test_conversations, output_dir / "test.jsonl")

    print("\nData preparation complete!")
    print(f"  Train: {len(train_conversations)} samples")
    print(f"  Val: {len(val_conversations)} samples")
    print(f"  Test: {len(test_conversations)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare medical literature data for fine-tuning")
    parser.add_argument("--train-samples", type=int, default=800, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    prepare_data(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        output_dir=args.output_dir
    )

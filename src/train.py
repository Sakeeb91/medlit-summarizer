"""
Training script for Medical Literature Summarizer using Tinker API.
Fine-tunes a model on PubMed summarization data.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import tinker
from tinker import types


# Configuration
BASE_MODEL = "Qwen/Qwen3-8B"  # Cost-effective model for summarization
LORA_RANK = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
CHECKPOINT_EVERY = 50  # Save checkpoint every N steps


def load_conversations(filepath: str) -> List[Dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(filepath, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def conversation_to_prompt(conversation: Dict, tokenizer) -> str:
    """Convert conversation to a formatted prompt string."""
    messages = conversation["messages"]

    # Build the full prompt
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return "\n".join(prompt_parts)


def prepare_datum(conversation: Dict, tokenizer) -> types.Datum:
    """
    Prepare a single Datum for training.

    The model learns to predict the assistant's response given the context.
    """
    messages = conversation["messages"]

    # Build context (system + user messages)
    context_parts = []
    assistant_content = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            context_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            context_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            context_parts.append(f"<|im_start|>assistant\n")
            assistant_content = content + "<|im_end|>"

    context = "\n".join(context_parts)

    # Tokenize
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    completion_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)

    # Full sequence
    all_tokens = context_tokens + completion_tokens

    # Create weights: 0 for context, 1 for completion
    weights = [0.0] * len(context_tokens) + [1.0] * len(completion_tokens)

    # Create Datum
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "weights": weights[1:],  # Shifted by 1 for next-token prediction
            "target_tokens": all_tokens[1:]
        }
    )

    return datum


def train(
    train_data_path: str = "data/train.jsonl",
    val_data_path: str = "data/val.jsonl",
    output_dir: str = "outputs",
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE
):
    """
    Main training function.

    Args:
        train_data_path: Path to training data JSONL
        val_data_path: Path to validation data JSONL
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Mini-batch size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Medical Literature Summarizer - Training")
    print("=" * 60)

    # Initialize Tinker client
    print("\nInitializing Tinker client...")
    service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK
    )

    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA rank: {LORA_RANK}")

    # Get tokenizer
    tokenizer = training_client.get_tokenizer()

    # Load data
    print("\nLoading training data...")
    train_conversations = load_conversations(train_data_path)
    val_conversations = load_conversations(val_data_path)
    print(f"  Train samples: {len(train_conversations)}")
    print(f"  Val samples: {len(val_conversations)}")

    # Prepare training data
    print("\nPreparing training data...")
    train_data = []
    for conv in tqdm(train_conversations, desc="Preparing train"):
        try:
            datum = prepare_datum(conv, tokenizer)
            train_data.append(datum)
        except Exception as e:
            print(f"  Skipping sample due to error: {e}")

    val_data = []
    for conv in tqdm(val_conversations, desc="Preparing val"):
        try:
            datum = prepare_datum(conv, tokenizer)
            val_data.append(datum)
        except Exception as e:
            print(f"  Skipping sample due to error: {e}")

    print(f"\nPrepared {len(train_data)} train, {len(val_data)} val samples")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    metrics_log = []
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Shuffle training data
        random.shuffle(train_data)

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = train_data[i:i + batch_size]

            if len(batch) == 0:
                continue

            # Forward-backward pass
            fwdbwd_future = training_client.forward_backward(
                batch,
                "cross_entropy"
            )

            # Optimizer step
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=learning_rate)
            )

            # Get results
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()

            # Track metrics
            batch_loss = fwdbwd_result.loss if hasattr(fwdbwd_result, 'loss') else 0.0
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            # Checkpoint
            if global_step % CHECKPOINT_EVERY == 0:
                checkpoint_name = f"step_{global_step:06d}"
                checkpoint_path = training_client.save_state(name=checkpoint_name).result().path
                print(f"\n  Checkpoint saved: {checkpoint_name}")

                metrics_log.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": batch_loss,
                    "checkpoint": checkpoint_path
                })

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        # Validation
        print("  Running validation...")
        val_loss = 0.0
        for batch_start in range(0, min(len(val_data), 50), batch_size):
            batch = val_data[batch_start:batch_start + batch_size]
            if len(batch) == 0:
                continue
            fwdbwd_result = training_client.forward_backward(
                batch,
                "cross_entropy"
            ).result()
            val_loss += fwdbwd_result.loss if hasattr(fwdbwd_result, 'loss') else 0.0

        val_loss = val_loss / max(min(len(val_data), 50) // batch_size, 1)
        print(f"  Validation loss: {val_loss:.4f}")

    # Final checkpoint
    print("\n" + "=" * 60)
    print("Training complete! Saving final model...")
    print("=" * 60)

    final_checkpoint = training_client.save_state(name="final").result().path
    print(f"  Final checkpoint: {final_checkpoint}")

    # Save sampling weights
    sampling_client = training_client.save_weights_and_get_sampling_client(name="final_sampler")
    sampling_path = f"tinker://medical-summarizer/final_sampler"

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "base_model": BASE_MODEL,
                "lora_rank": LORA_RANK,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs
            },
            "final_checkpoint": final_checkpoint,
            "metrics": metrics_log
        }, f, indent=2)

    print(f"\n  Metrics saved to: {metrics_path}")
    print("\nTraining complete!")

    return training_client, sampling_client


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Medical Literature Summarizer")
    parser.add_argument("--train-data", type=str, default="data/train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/val.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

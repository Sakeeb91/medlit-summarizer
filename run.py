#!/usr/bin/env python3
"""
Medical Literature Summarizer - Main Entry Point

This script provides a simple interface to:
1. Prepare training data from PubMed
2. Fine-tune a model using Tinker
3. Run inference on medical articles

Usage:
    python run.py prepare   - Download and prepare PubMed data
    python run.py train     - Fine-tune the model
    python run.py summarize - Summarize a medical article
    python run.py demo      - Run a demo with sample article
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def prepare_data():
    """Download and prepare PubMed summarization dataset."""
    from data_preparation import prepare_data as _prepare

    print("Preparing PubMed summarization data...")
    _prepare(
        train_samples=500,   # Adjust based on your needs
        val_samples=50,
        test_samples=50,
        output_dir="data"
    )


def train_model():
    """Fine-tune the model on medical literature."""
    from train import train

    print("Starting training...")
    train(
        train_data_path="data/train.jsonl",
        val_data_path="data/val.jsonl",
        output_dir="outputs",
        num_epochs=2,
        learning_rate=1e-4,
        batch_size=4
    )


def summarize_article(article_path: str = None, model_path: str = None):
    """Summarize a medical article."""
    from summarize import MedicalSummarizer

    summarizer = MedicalSummarizer(
        model_path=model_path,
        use_base_model=(model_path is None)
    )

    if article_path:
        with open(article_path, "r") as f:
            article = f.read()
    else:
        print("Enter article text (Ctrl+D when done):")
        article = sys.stdin.read()

    print("\nGenerating summary...")
    summary = summarizer.summarize(article)

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(summary)


def run_demo():
    """Run a demo with a sample medical article."""
    from summarize import MedicalSummarizer

    # Sample medical article abstract for demo
    sample_article = """
    Background: Cardiovascular disease remains the leading cause of mortality worldwide.
    Recent studies have suggested that novel biomarkers may improve risk stratification
    beyond traditional risk factors. This study aimed to evaluate the prognostic value
    of high-sensitivity cardiac troponin T (hs-cTnT) in a general population cohort.

    Methods: We conducted a prospective cohort study of 8,121 participants from the
    community-based ARIC study (Atherosclerosis Risk in Communities). Participants
    were free of cardiovascular disease at baseline (1990-1992) and were followed for
    incident cardiovascular events through 2015. hs-cTnT was measured in stored plasma
    samples using a high-sensitivity assay. Cox proportional hazards models were used
    to examine associations with cardiovascular outcomes.

    Results: During a median follow-up of 20.6 years, 1,821 participants developed
    cardiovascular disease, 876 developed heart failure, and 612 experienced coronary
    heart disease events. After adjustment for traditional risk factors, participants
    in the highest quartile of hs-cTnT had a 2.3-fold increased risk of cardiovascular
    disease (HR 2.31, 95% CI 1.98-2.70), a 4.2-fold increased risk of heart failure
    (HR 4.23, 95% CI 3.31-5.40), and a 1.8-fold increased risk of coronary heart disease
    (HR 1.82, 95% CI 1.43-2.32) compared with participants in the lowest quartile.
    Addition of hs-cTnT to traditional risk factors improved discrimination (C-statistic
    increase: 0.023 for CVD, 0.051 for heart failure) and reclassification metrics.

    Conclusions: hs-cTnT measured in community-dwelling adults predicts long-term risk
    of cardiovascular disease, particularly heart failure, independent of traditional
    risk factors. These findings support the potential utility of hs-cTnT for cardiovascular
    risk assessment in primary prevention settings.

    Keywords: cardiac troponin, biomarkers, cardiovascular disease, heart failure, risk prediction
    """

    print("=" * 60)
    print("Medical Literature Summarizer - Demo")
    print("=" * 60)
    print("\nSample Article:")
    print("-" * 40)
    print(sample_article[:500] + "...")
    print("-" * 40)

    print("\nInitializing summarizer (using base model for demo)...")
    summarizer = MedicalSummarizer(use_base_model=True)

    print("\nGenerating summary...")
    summary = summarizer.summarize(sample_article)

    print("\n" + "=" * 50)
    print("GENERATED SUMMARY:")
    print("=" * 50)
    print(summary)
    print("=" * 50)


def interactive_mode():
    """Run interactive summarization mode."""
    from summarize import MedicalSummarizer, interactive_mode

    print("Starting interactive mode...")
    summarizer = MedicalSummarizer(use_base_model=True)
    interactive_mode(summarizer)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    # Set API key if not already set
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Warning: TINKER_API_KEY not set. Please set it:")
        print("  export TINKER_API_KEY=your_api_key")
        print()

    if command == "prepare":
        prepare_data()
    elif command == "train":
        train_model()
    elif command == "summarize":
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        article_path = sys.argv[3] if len(sys.argv) > 3 else None
        summarize_article(article_path, model_path)
    elif command == "demo":
        run_demo()
    elif command == "interactive":
        interactive_mode()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

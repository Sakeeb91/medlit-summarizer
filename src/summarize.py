"""
Inference script for Medical Literature Summarizer.
Summarize medical research articles using the fine-tuned model.
"""

import os
from typing import Optional

import tinker
from tinker import types
from openai import OpenAI


SYSTEM_PROMPT = """You are an expert medical literature summarizer. Your task is to read medical research articles and produce clear, accurate, and concise summaries that capture the key findings, methodology, and conclusions. Focus on:
- Main research objectives and hypotheses
- Key methodology and study design
- Important results and statistical findings
- Clinical implications and conclusions

Maintain scientific accuracy while making the summary accessible."""


class MedicalSummarizer:
    """Medical Literature Summarizer using fine-tuned Tinker model."""

    def __init__(self, model_path: Optional[str] = None, use_base_model: bool = False):
        """
        Initialize the summarizer.

        Args:
            model_path: Path to fine-tuned model weights (tinker://...)
                       If None, uses the base model for inference
            use_base_model: If True, uses base model without fine-tuning
        """
        self.service_client = tinker.ServiceClient()

        if use_base_model or model_path is None:
            # Use base model
            self.base_model = "Qwen/Qwen3-8B"
            training_client = self.service_client.create_lora_training_client(
                base_model=self.base_model
            )
            self.sampling_client = training_client.save_weights_and_get_sampling_client(
                name="base"
            )
            self.tokenizer = training_client.get_tokenizer()
            print(f"Using base model: {self.base_model}")
        else:
            # Use fine-tuned model
            self.sampling_client = self.service_client.create_sampling_client(
                model_path=model_path
            )
            # Get tokenizer from a training client
            training_client = self.service_client.create_lora_training_client(
                base_model="Qwen/Qwen3-8B"
            )
            self.tokenizer = training_client.get_tokenizer()
            print(f"Using fine-tuned model: {model_path}")

    def summarize(
        self,
        article: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> str:
        """
        Summarize a medical research article.

        Args:
            article: The medical article text to summarize
            max_tokens: Maximum tokens in the summary
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Top-p sampling parameter

        Returns:
            Summary of the article
        """
        # Truncate very long articles
        article = article[:12000]

        # Build prompt (using /no_think for direct responses without chain-of-thought)
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Please summarize the following medical research article: /no_think

{article}<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        # Sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        # Generate
        result = self.sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1
        ).result()

        # Decode response
        if result.sequences:
            output_tokens = result.sequences[0].tokens
            summary = self.tokenizer.decode(output_tokens)
            return summary.strip()
        else:
            return "Error: No summary generated"

    def summarize_batch(
        self,
        articles: list,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> list:
        """
        Summarize multiple articles.

        Args:
            articles: List of article texts
            max_tokens: Maximum tokens per summary
            temperature: Sampling temperature

        Returns:
            List of summaries
        """
        summaries = []
        for i, article in enumerate(articles):
            print(f"Summarizing article {i + 1}/{len(articles)}...")
            summary = self.summarize(article, max_tokens, temperature)
            summaries.append(summary)
        return summaries


class MedicalSummarizerOpenAI:
    """
    Medical Literature Summarizer using OpenAI-compatible API.
    Useful for testing fine-tuned models via the compatible endpoint.
    """

    def __init__(self, model_path: str, api_key: Optional[str] = None):
        """
        Initialize the OpenAI-compatible summarizer.

        Args:
            model_path: Tinker model path (tinker://...)
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
        self.api_key = api_key or os.environ.get("TINKER_API_KEY")
        self.model_path = model_path

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        print(f"Using OpenAI-compatible API with model: {model_path}")

    def summarize(
        self,
        article: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """
        Summarize a medical research article.

        Args:
            article: The medical article text to summarize
            max_tokens: Maximum tokens in the summary
            temperature: Sampling temperature

        Returns:
            Summary of the article
        """
        # Truncate very long articles
        article = article[:12000]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please summarize the following medical research article:\n\n{article}"}
        ]

        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content


def interactive_mode(summarizer):
    """Run interactive summarization mode."""
    print("\n" + "=" * 60)
    print("Medical Literature Summarizer - Interactive Mode")
    print("=" * 60)
    print("\nPaste your medical article text and press Enter twice to summarize.")
    print("Type 'quit' to exit.\n")

    while True:
        print("-" * 40)
        print("Enter article text (or 'quit' to exit):")

        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                print("Goodbye!")
                return
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        article = "\n".join(lines[:-1])  # Remove trailing empty line

        if not article.strip():
            print("No text provided. Please try again.")
            continue

        print("\nGenerating summary...")
        summary = summarizer.summarize(article)

        print("\n" + "=" * 40)
        print("SUMMARY:")
        print("=" * 40)
        print(summary)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize medical literature")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to fine-tuned model (tinker://...)")
    parser.add_argument("--base-model", action="store_true",
                       help="Use base model without fine-tuning")
    parser.add_argument("--article", type=str, default=None,
                       help="Article text to summarize (or path to file)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Maximum tokens in summary")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Sampling temperature")

    args = parser.parse_args()

    # Initialize summarizer
    summarizer = MedicalSummarizer(
        model_path=args.model_path,
        use_base_model=args.base_model
    )

    if args.interactive:
        interactive_mode(summarizer)
    elif args.article:
        # Check if it's a file path
        if os.path.exists(args.article):
            with open(args.article, "r") as f:
                article = f.read()
        else:
            article = args.article

        summary = summarizer.summarize(
            article,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        print("\nSUMMARY:")
        print("=" * 40)
        print(summary)
    else:
        print("Please provide --article or --interactive flag")
        parser.print_help()

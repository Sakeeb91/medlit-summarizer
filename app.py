"""
Flask backend for Medical Literature Summarizer.
Provides API endpoints for summarizing medical articles.
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tinker
from tinker import types

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
MODEL_CHECKPOINT = "tinker://9777df8e-eb3e-524a-9867-47a14d63af46:train:0/weights/final"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Instruction-tuned model for better task following

SYSTEM_PROMPT = """You are an expert medical literature summarizer. Your task is to read medical research articles and produce clear, accurate, and concise summaries that capture the key findings, methodology, and conclusions. Focus on:
- Main research objectives and hypotheses
- Key methodology and study design
- Important results and statistical findings
- Clinical implications and conclusions

Maintain scientific accuracy while making the summary accessible."""

# Global clients (initialized on first request)
_service_client = None
_sampling_client = None
_tokenizer = None


def get_clients():
    """Initialize and return Tinker clients."""
    global _service_client, _sampling_client, _tokenizer

    if _sampling_client is None:
        print("Initializing Tinker clients...")
        _service_client = tinker.ServiceClient()

        training_client = _service_client.create_lora_training_client(
            base_model=BASE_MODEL
        )

        # Use base model directly (it's already instruction-tuned)
        print(f"Using base model: {BASE_MODEL}")

        # Get sampling client and tokenizer
        _sampling_client = training_client.save_weights_and_get_sampling_client(
            name="web_inference"
        )
        _tokenizer = training_client.get_tokenizer()
        print("Clients initialized successfully!")

    return _sampling_client, _tokenizer


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Summarize a medical article."""
    try:
        data = request.json
        article = data.get('article', '')

        if not article.strip():
            return jsonify({'error': 'No article text provided'}), 400

        # Truncate very long articles
        article = article[:12000]

        # Get clients
        sampling_client, tokenizer = get_clients()

        # Build prompt
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Please summarize the following medical research article: /no_think

{article}<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        # Sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=600,
            temperature=0.3,
            top_p=0.9,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        # Generate summary
        result = sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1
        ).result()

        if result.sequences:
            output_tokens = result.sequences[0].tokens
            summary = tokenizer.decode(output_tokens)

            # Clean up the summary
            summary = summary.strip()

            # Remove thinking tags if present
            if summary.startswith('<think>'):
                think_end = summary.find('</think>')
                if think_end != -1:
                    summary = summary[think_end + 8:].strip()

            # Remove any trailing special tokens
            for token in ['<|im_end|>', '<|im_start|>', '<|endoftext|>']:
                summary = summary.replace(token, '').strip()

            return jsonify({
                'success': True,
                'summary': summary,
                'article_length': len(article),
                'summary_length': len(summary)
            })
        else:
            return jsonify({'error': 'No summary generated'}), 500

    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': BASE_MODEL})


if __name__ == '__main__':
    print("=" * 60)
    print("Medical Literature Summarizer - Web Interface")
    print("=" * 60)
    print(f"Model: {BASE_MODEL}")
    print(f"Checkpoint: {MODEL_CHECKPOINT}")
    print()
    print("Starting server at http://localhost:8080")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8080, debug=False)

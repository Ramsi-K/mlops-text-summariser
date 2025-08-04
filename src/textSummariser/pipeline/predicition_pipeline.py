from transformers import AutoTokenizer, pipeline

from src.textSummariser.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        import os
        import time

        start_time = time.time()

        # Check if fine-tuned model exists, otherwise use base model
        if os.path.exists(self.config.model_path) and os.path.exists(
            self.config.tokenizer_path
        ):
            model_path = self.config.model_path
            tokenizer_path = self.config.tokenizer_path
            print(f"Using fine-tuned model from {model_path}")
        else:
            # Fallback to base model
            model_path = "google/pegasus-cnn_dailymail"
            tokenizer_path = "google/pegasus-cnn_dailymail"
            print(
                "Fine-tuned model not found, using base model: google/pegasus-cnn_dailymail"
            )

        try:
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            print("Creating pipeline...")
            gen_kwargs = {
                "length_penalty": 0.8,
                "num_beams": 8,
                "max_length": 128,
            }
            pipe = pipeline("summarization", model=model_path, tokenizer=tokenizer)

            print("Dialogue:")
            print(text)

            print("Generating summary...")
            output = pipe(text, **gen_kwargs)[0]["summary_text"]

            elapsed_time = time.time() - start_time
            print(f"\nModel Summary (generated in {elapsed_time: .2f}s): ")
            print(output)

            return output

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Return a simple fallback summary
            return f"Summary generation failed. Original text length: {len(text)} characters. Error: {str(e)}"

from typing import Optional

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers.pipelines.base import Pipeline


class LlamaRunner:
    """
    Class which can be used to load a llama-architecture model (and optionally a LoRA model with it) and perform
    inference using it.
    """

    def __init__(self, model_id: str, lora_id: Optional[str]):
        """
        Initializes the llama runner with the specified model.
        :param model_id: huggingface model id or file path to model.
        :param lora_id: If not None, the huggingface lora model id or file path to lora model. If None, no LoRA model
        is loaded.
        """
        self.model_id = model_id

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=quantization_config)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

        if lora_id is not None:
            model = PeftModel.from_pretrained(model, lora_id).merge_and_unload()

        self.pipeline: Pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        self.__hyperparameters = {
            "temperature": 0.7,
            "top_p": 0.1,
            "top_k": 40,
            "typical_p": 1,
            "min_p": 0,
            "repetition_penalty": 1.1
        }

    def generate_text(self, prompt: str, answer_schema=None, max_new_tokens: int = 3000) -> str:
        if answer_schema is not None:
            parser = JsonSchemaParser(answer_schema)
            prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipeline.tokenizer, parser)
            result = self.pipeline(prompt,
                                   prefix_allowed_tokens_fn=prefix_function,
                                   max_new_tokens=max_new_tokens,
                                   **self.__hyperparameters)
        else:
            result = self.pipeline(prompt,
                                   max_new_tokens=max_new_tokens,
                                   **self.__hyperparameters)

        return result[0]["generated_text"]


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_id", type=str, required=True, help="Huggingface model id or file path to model.")
    arg_parser.add_argument("--lora_id", type=str, required=True, help="Huggingface lora model id or file path to lora model.")
    arg_parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate text from.")

    args = arg_parser.parse_args()

    runner = LlamaRunner(args.model_id, args.lora_id)
    print(runner.generate_text(args.prompt))
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pydantic
from transformers.pipelines.base import Pipeline
from peft import PeftModel


class LlamaRunner:
    def __init__(self, model_id: str, lora_id: str):
        """
        Initializes the llama runner with the specified model.
        :param model_id: huggingface model id or file path to model.
        :param lora_id: huggingface lora model id or file path to lora model.
        """
        self.model_id = model_id

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=quantization_config)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        model = PeftModel.from_pretrained(model, lora_id).merge_and_unload()

        self.pipeline: Pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def generate_text(self, prompt: str, answer_schema=None) -> str:
        if answer_schema is not None:
            parser = JsonSchemaParser(answer_schema)
            prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipeline.tokenizer, parser)
            result = self.pipeline(prompt, prefix_allowed_tokens_fn=prefix_function, max_new_tokens=10000)
        else:
            result = self.pipeline(prompt, max_new_tokens=500)

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
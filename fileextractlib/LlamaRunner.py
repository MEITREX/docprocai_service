from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pydantic
from transformers.pipelines.base import Pipeline


class LlamaRunner:
    def __init_(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        self.model_id = model_id

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=quantization_config)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

        self.pipeline: Pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def generate_text(self, prompt: str, answer_schema=None):
        if answer_schema != None:
            parser = JsonSchemaParser(answer_schema.model_json_schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipeline.tokenizer, parser)

        result = self.pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)

        return result
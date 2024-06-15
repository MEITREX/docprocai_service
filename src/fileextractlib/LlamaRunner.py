from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pydantic

model_id = "meta-llama/Meta-Llama-3-8B"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

#tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=quantization_config)
#model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

#pipeline = pipeline(
#    "text-generation",
#    model=model,
#    tokenizer=tokenizer,
#    model_kwargs={"torch_dtype": torch.bfloat16},
#)

def generate_text(prompt: str, answer_schema=None):
    if answer_schema != None:
        parser = JsonSchemaParser(answer_schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    result = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)

    return result
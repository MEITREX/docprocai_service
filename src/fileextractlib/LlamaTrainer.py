import peft
import transformers
import math
import datasets

lora_rank = 512
lora_alpha = 64
lora_dropout = 0.1
micro_batch_size = 1
gradient_accumulation_steps = 1
warmup_steps = 200
training_epochs = 3
learning_rate = 3e-4

output_path = "output"

model_id = "meta-llama/Meta-Llama-3-8B"

bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
base_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

lora_modules = peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["llama"]

lora_config = peft.LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=lora_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

lora_model = peft.get_peft_model(base_model, lora_config)

class TrainingCallbacks(transformers.TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']:.4f}")

trainer = transformers.Trainer(
    model=lora_model,
    train_dataset=None, # TODO
    eval_dataset=None, # TODO
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=training_epochs,
        learning_rate=learning_rate,
        fp16=True,
        optim="adamw",
        logging_steps = 1,
        evaluation_strategy="no", # can use "steps" if we pass some eval dataset
        eval_steps=10,
        save_strategy="no",
        output_dir=output_path,
        use_ipex=True if transformers.is_torch_xpu_available() else False
    ),
    callbacks=list(TrainingCallbacks())
)

trainer.train()

lora_model.save_pretrained(output_path)
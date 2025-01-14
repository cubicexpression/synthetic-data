import sys
import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

################
# Model Loading
################
checkpoint_path = "microsoft/Phi-3.5-mini-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
csv_path = "./fine_tune_data.csv"  # Path to your dataset
df = pd.read_csv(csv_path)

if not {"transcript", "reference"}.issubset(df.columns):
    raise ValueError("Dataset must include 'transcript' and 'reference' columns.")

system_prompt = (
    "You are a fraud prevention assistant helping advisors handle customer queries.\n"
)

def format_data(example):
    transcript = f"**Transcript:**\n{str(example['transcript'])}"
    reference = str(example['reference'])
    formatted_input = f"{system_prompt}{transcript}"
    return {"messages": [{"role": "user", "content": formatted_input}], "reference": reference}

# Format and convert to Dataset
formatted_data = df.apply(format_data, axis=1).to_list()
dataset = Dataset.from_list(formatted_data)

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# Apply template
def apply_chat_template(example):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example

train_dataset = dataset_dict["train"].map(
    apply_chat_template,
    num_proc=4,
    remove_columns=list(dataset_dict["train"].features),
    desc="Applying chat template to train dataset",
)

test_dataset = dataset_dict["test"].map(
    apply_chat_template,
    num_proc=4,
    remove_columns=list(dataset_dict["test"].features),
    desc="Applying chat template to test dataset",
)

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
metrics = trainer.evaluate()
metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(train_conf.output_dir)

print("Fine-tuning completed. Model saved at './checkpoint_dir'.")

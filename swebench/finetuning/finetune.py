import os
# Set PyTorch CUDA memory allocation to expand segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig
)


from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


bnb_config = BitsAndBytesConfig(
    #load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16,    
)

# Load the model
#model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-8B"
#model_name = "llama"

# Check GPU memory before loading
print(f"Before tokenizer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Fix: Don't set padding='max_length' here, let the data collator handle it
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    revision="main",
    use_fast=True,
#    truncation=True,
#    max_length=40000,
#    padding='max_length', #'max_length',
)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

print(f"After tokenizer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())


# Load base model with 4-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Explicitly set to float16
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    revision="main"
)

model.config.use_cache = False # Disable caching to save memory during training, set to True for inference
model.config.pretraining_tp = 1 # is a configuration parameter for the model, where tp stands for tensor parallelism.
# Setting pretraining_tp = 1 means no tensor parallelism (i.e., the model will not split its weights across multiple devices for parallel computation).


print(f"After AutoModelForCausalLM: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

print(f"After prepare_model_for_kbit_training: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())


# Configure LoRA
peft_config = LoraConfig(
    r=64,                   # Rank
    lora_alpha=16,# 16,         # Alpha parameter
#    target_modules=["q_proj", "v_proj"],
#    target_modules=["q_proj", "v_proj"],#, "k_proj", "o_proj"],
#    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
#                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,     # Dropout probability
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    ],
)
# Apply LoRA to model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print(f"After get_peft_model: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())


from datasets import load_dataset
from swebench.inference.make_datasets.tokenize_dataset import main as tokenize_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("princeton-nlp/SWE-bench_bm25_40K")
#ds = load_dataset("princeton-nlp/SWE-bench_bm25_13K")
ds = ds.rename_column("text", "prompt")
ds = ds.rename_column("patch", "completion")
ds = ds.remove_columns([col for col in ds['train'].column_names if col not in ["prompt", "completion"]])


def tokenize_function(examples):
    #print(len(examples["prompt"]))
    # Tokenize prompts and completions separately
    tokenized_prompts = tokenizer(
        ['<|im_start|>' + x + '<|im_end|>' for x in examples["prompt"]],
        truncation=False, #True,
        #max_length=3072,  # Adjust based on your needs
        padding=False,
        return_tensors=None,
    )
    
    tokenized_completions = tokenizer(
        ['<|im_start|>' + x + '<|im_end|>' for x in examples["completion"]],
        truncation=False, #True,
        #max_length=1024,  # Adjust based on your needs
        padding=False,
        return_tensors=None,
    )
    
    # Create separate input_ids field for both prompt and completion
    result = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for i in range(len(tokenized_prompts["input_ids"])):
        # Combine prompt and completion input_ids
        combined_input_ids = tokenized_prompts["input_ids"][i] + tokenized_completions["input_ids"][i]
        combined_attention_mask = tokenized_prompts["attention_mask"][i] + tokenized_completions["attention_mask"][i]
        
        # For labels, use -100 for prompt tokens (to ignore them in loss) and actual token ids for completion
        labels = [-100] * len(tokenized_prompts["input_ids"][i]) + tokenized_completions["input_ids"][i]
        
        result["input_ids"].append(combined_input_ids)
        result["attention_mask"].append(combined_attention_mask)
        result["labels"].append(labels)
    
    return result

if False:
    if False:
        # Process dataset in smaller batches to avoid OOM
        tokenized_ds = ds.map(
            tokenize_function,
            batched=True,
            batch_size=16,  # Process in small batches
            remove_columns=["prompt", "completion"],  # Remove original text columns
            num_proc=4,  # Use multiple processes
            desc="Tokenizing dataset with completion-only labels",
        )

        tokenized_ds.save_to_disk("./tokenized_swe_bench_bm25_40K")
    else:
        tokenized_ds = load_from_disk("./tokenized_swe_bench_bm25_40K")

    def filter_by_length(examples, max_length):
        return  [len(ids) <= max_length for ids in examples["input_ids"]]

    tokenized_short_ds = tokenized_ds
    for max_length in [15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 41_000, 42_000, 43_000, 44_000, 45_000, 50_000][::-1]:
    #for max_length in [42_500]:
        print(f'-------- Max length: {max_length} --------')
        tokenized_short_ds = tokenized_short_ds.filter(filter_by_length, 
                                    fn_kwargs={"max_length":max_length},  # Pass your desired max length
                                    num_proc=4,
                                    batch_size=16,
                                    batched=True,
                                    desc=f"Filtering keeping only sequences shorter than {max_length} tokens")

        print('\tNumber of rows (train): ', tokenized_short_ds["train"].num_rows)
        print('\tNumber of rows (validation): ', tokenized_short_ds["validation"].num_rows)

        tokenized_short_ds.save_to_disk(f"./tokenized_swe_bench_bm25_40K_short_{max_length}")
else:
    max_length = 44_000
    mapped_ds = load_from_disk(f"./tokenized_swe_bench_bm25_40K_short_{max_length}", keep_in_memory=False)

#max_seq_length = 40000

# Fix: Use an appropriate data collator that handles padding properly
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Fix: Update SFTConfig to correctly handle variable length sequences
sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduce batch size to save memory
    gradient_accumulation_steps=8,  # Increase gradient accumulation to compensate
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=6e-4,
    fp16=True,
    remove_unused_columns=True,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    optim="adamw_torch",  # Use memory-efficient optimizer
    max_grad_norm=0.3,    # Clip gradients to prevent spikes
    completion_only_loss=True,
    disable_tqdm=False,
)   

print(f"After data_collator: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    peft_config=peft_config,
#    train_dataset=ds["train"].select(range(5)),
#    eval_dataset=ds["validation"].select(range(5)),
#    data_collator=data_collator,
    train_dataset=mapped_ds["train"],
    eval_dataset=mapped_ds["validation"],
)


print(f"After SFTTrainer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())


import gc, torch
gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False


# Create a custom callback to periodically clear CUDA cache
class CacheClearingCallback(TrainerCallback):
    def __init__(self, steps_interval=50):
        self.steps_interval = steps_interval
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps_interval == 0:
            torch.cuda.empty_cache()
            print(f"Step {state.global_step}: Cleared CUDA cache")

# Add the callback to periodically clear cache
trainer.add_callback(CacheClearingCallback(steps_interval=5))


print(f"After SFTTrainer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

trainer.train()

print(f"After trainer.train: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Save the model adapter
#model.save_pretrained("./qwen3-swe-bench-bm25_40K-lora-epoch4")
model.save_pretrained("./qwen3-swe-bench-bm25_40K-lora-epoch1-8B-64r-16a-44000")

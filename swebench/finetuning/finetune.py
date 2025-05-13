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
    prepare_model_for_kbit_training,∏
    TaskType
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model
model_name = "Qwen/Qwen3-0.6B"
#model_name = "llama"

# Check GPU memory before loading
print(f"Before tokenizer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    attn_implementation="flash_attention_2",
)

model.config.use_cache = False # Disable caching to save memory during training, set to True for inference
#model.config.pretraining_tp = 1


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

if False:
    # Login using e.g. `huggingface-cli login` to access this dataset
    #ds = load_dataset("SWE-bench/SWE-bench")
    ds = load_dataset("princeton-nlp/SWE-bench_oracle")


    # Tokenize the dataset
    tokenize_dataset(
        dataset_name_or_path='./training_data',
        output_dir="./tokenized_data",
        tokenizer_name="llama",  # Using llama tokenizer
        num_proc=30,
        push_to_hub_user=None
    )

    # Now tokenize the dataset
    ds.save_to_disk('./training_data')


    # https://chatgpt.com/c/681a65f3-8a18-800e-b2b9-1fb183a4dab4

# Load the tokenized dataset
tokenized_dataset = load_from_disk("./tokenized_data/training_data__tok-llama")

print(f"After tokenized_dataset: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

df_train = tokenized_dataset["train"].to_pandas()

print(f"After df_train: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Function to determine max length for a batch
def get_max_length_in_dataset(dataset, percentile=95):
    lengths = [len(x["input_ids"]) for x in dataset]
    import numpy as np
    return int(np.percentile(lengths, percentile))

# Get a reasonable max length instead of using the longest sequence
#max_seq_length = get_max_length_in_dataset(tokenized_dataset["train"])
import numpy as np
max_seq_length = int(np.percentile([x["input_ids"].shape[0] for _, x in df_train.iterrows() ], 95))
print(f"Using max sequence length: {max_seq_length}")
print(f"After max_seq_length: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=5,  # Reduce batch size to save memory
    gradient_accumulation_steps=16,  # Increase gradient accumulation to compensate
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    remove_unused_columns=True,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    optim="adamw_torch",  # Use memory-efficient optimizer
    max_grad_norm=0.3,    # Clip gradients to prevent spikes
)

# Create a data collator with memory-efficient padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8
)
print(f"After data_collator: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Define a memory-efficient collate function
def custom_data_collator(features):
    ## Pre-calculate max length to minimize recomputation
    #max_length = min(
    #    max(len(feature["input_ids"]) for feature in features),
    #    max_seq_length
    #)
    max_length = 4096 #     max_seq_length
    # Pre-allocate tensors with the right size
    batch_size = len(features)
    input_ids = torch.full((batch_size, max_length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    # Fill tensors without creating intermediate tensors
    for i, feature in enumerate(features):
        seq_length = min(len(feature["input_ids"]), max_length)
        # Avoid creating temporary tensors where possible
        if isinstance(feature["input_ids"], torch.Tensor):
            input_ids[i, :seq_length] = feature["input_ids"][:seq_length]
            labels[i, :seq_length] = feature["labels"][:seq_length]
        else:
            input_ids[i, :seq_length] = torch.tensor(feature["input_ids"][:seq_length], dtype=torch.long)
            labels[i, :seq_length] = torch.tensor(feature["labels"][:seq_length], dtype=torch.long)
        attention_mask[i, :seq_length] = 1
    
    # Let the trainer handle device placement
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],#.select(range(2)),
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    data_collator=custom_data_collator,
)
print(f"After trainer: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Empty CUDA cache before training
torch.cuda.empty_cache()

print(f"After torch.cuda.empty_cache: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Create a custom callback to periodically clear CUDA cache
class CacheClearingCallback(TrainerCallback):
    def __init__(self, steps_interval=50):
        self.steps_interval = steps_interval
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps_interval == 0:
            torch.cuda.empty_cache()
            print(f"Step {state.global_step}: Cleared CUDA cache")

# Add the callback to periodically clear cache
trainer.add_callback(CacheClearingCallback(steps_interval=50))
torch.cuda.empty_cache()

# Train the model
trainer.train()
print(f"After trainer.train: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(torch.cuda.memory_summary())

# Save the model adapter
model.save_pretrained("./qwen3-swe-bench-lora-epoch2")


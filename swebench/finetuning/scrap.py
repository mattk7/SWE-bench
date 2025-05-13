
""" def prepare_dataset_for_training(examples):
    ""Prepare the dataset by combining text and patch into the format needed for training.""
    # Combine the text and patch with a separator
    combined_text = [f"{text}\n{patch}" for text, patch in zip(examples["text"], examples["patch"])]
    
    # Tokenize the combined text
    tokenized = tokenizer(
        combined_text,
        padding="max_length",
        truncation=True,
        max_length=32768,  # Adjust based on your model's context window
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Prepare the datasets
train_dataset = tokenized_dataset["train"].map(
    prepare_dataset_for_training,
    batched=True,
    remove_columns=tokenized_dataset["train"].column_names  # Remove original columns
)

eval_dataset = None
if "validation" in tokenized_dataset:
    eval_dataset = tokenized_dataset["validation"].map(
        prepare_dataset_for_training,
        batched=True,
        remove_columns=tokenized_dataset["validation"].column_names
    )
 """




# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].map(
        lambda x: {
            "input_ids": x["input_ids"][:2048],  # Truncate to max length
            "labels": x["labels"][:2048],  # Truncate to max length
        },
        remove_columns=[col for col in tokenized_dataset["train"].column_names if col not in ["input_ids", "labels"]]
    ),
    eval_dataset=tokenized_dataset["validation"].map(
        lambda x: {
            "input_ids": x["input_ids"][:2048],  # Truncate to max length
            "labels": x["labels"][:2048],  # Truncate to max length
        },
        remove_columns=[col for col in tokenized_dataset["validation"].column_names if col not in ["input_ids", "labels"]]
    ) if "validation" in tokenized_dataset else None,
    data_collator=data_collator,
)

# Train the model
trainer.train()
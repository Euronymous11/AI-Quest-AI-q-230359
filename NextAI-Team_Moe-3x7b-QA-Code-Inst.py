from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch

def main():
    # Detect device and GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    # Load model and tokenizer
    model_name = "NextAI-Team/Moe-3x7b-QA-Code-Inst"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Enable gradient checkpointing after loading the model
    model.gradient_checkpointing_enable()  # Correct way to enable gradient checkpointing
    
    # Load datasets
    datasets_to_load = [
        "OpenCoder-LLM/opc-sft-stage1",
        "OpenCoder-LLM/fineweb-code-corpus",
        "glaiveai/glaive-function-calling-v2",
        "fine-tuned/askubuntu"
    ]
    datasets = [load_dataset(dataset, split="train") for dataset in datasets_to_load]
    combined_dataset = concatenate_datasets(datasets)

    # Preprocess data
    def preprocess_function(example):
        prompt = example.get("instruction", example.get("question", ""))
        response = example.get("response", example.get("answer", ""))
        inputs = tokenizer(
            prompt,
            text_pair=response,
            truncation=True,
            max_length=512,
            padding=False,  # No padding yet to save memory
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    processed_dataset = combined_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=combined_dataset.column_names
    )

    # Enable dynamic padding for efficient batch sizes
    def data_collator(features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {
            key: torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f[key]) for f in features],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            for key in features[0]
        }
        batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()
        return batch

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./moe-qa-code-t4-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="epoch",
        logging_strategy="steps",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Improved throughput
        fp16=True,  # Faster training with mixed precision
        num_train_epochs=3,
        warmup_steps=500,  # Gradual learning rate increase
        learning_rate=3e-5,  # Optimized learning rate for large models
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        ddp_find_unused_parameters=False,
        optim="adamw_torch",  # Use PyTorch's AdamW optimizer
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
    trainer.save_model("./moe-qa-code-t4-finetuned")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()

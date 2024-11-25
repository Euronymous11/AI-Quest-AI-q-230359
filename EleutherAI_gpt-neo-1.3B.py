from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

def main():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model and tokenizer
    model_name = "EleutherAI/gpt-neo-1.3B"  # Optimized for Colab
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model and move to GPU
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=4,  # Efficiency rank
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Load the entire dataset (non-streaming mode)
    dataset = load_dataset(
        "OpenCoder-LLM/opc-sft-stage1",
        "filtered_infinity_instruct",
        split="train"  # Loads all data into memory
    )

    # Preprocessing
    def preprocess_function(example):
        inputs = tokenizer(
            example["instruction"],
            truncation=True,
            padding="max_length",
            max_length=64,  # Adjusted for larger sequence processing
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction"])

    # Set training arguments optimized for Colab
    training_args = TrainingArguments(
        output_dir="./gpt-neo-lora-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,  # Fully iterate over the dataset
        per_device_train_batch_size=4,  # Reduce to fit Colab's GPU memory
        gradient_accumulation_steps=8,  # Compensates for smaller batch size
        fp16=True,  # Enable mixed precision
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model("./gpt-neo-lora-finetuned")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()

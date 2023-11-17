from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from dataset_preparation import load_hf_data, load_csv_data
import torch
from metrics import compute_metrics
from peft import LoraConfig, TaskType, get_peft_model

# setting the device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_title_gen(pretrained_model, batch_size, output_dir, lr, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Setting model and tokenizer")

    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if param_efficient:
        print("-----------------------")
        print("Using parameter efficient fine-tuning")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"
            ]
        )

        print("-----------------------")
        print("Getting PEFT model")

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("-----------------------")
    print("Loading data")

    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train.csv", "./data/atn-filtered-publication-section-val.csv", "./data/atn-filtered-publication-section-test.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False
    )

    print("-----------------------")
    print("Setting training arguments")

    # Set up Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    compute_metrics_lambda = lambda x: compute_metrics(tokenizer, x)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_lambda,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("-----------------------")
    print("Training the model")

    trainer.train()

# train_title_gen(
#     pretrained_model="facebook/bart-base", 
#     batch_size=32, 
#     output_dir="results/bart/doc2title/all", 
#     lr=8e-4, 
#     weight_decay=1e-1, 
#     num_epochs=5, 
#     chunk_size=1e4, 
#     nrows=None, 
#     filters=None,
#     param_efficient=True
# )

# train_title_gen(
#     pretrained_model="google/long-t5-local-base", 
#     batch_size=32, 
#     output_dir="results/long-t5/doc2title/all", 
#     lr=8e-4, 
#     weight_decay=1e-1, 
#     num_epochs=5, 
#     chunk_size=1e4, 
#     nrows=None, 
#     filters=None,
#     param_efficient=True
# )

train_title_gen(
    pretrained_model="facebook/bart-base", 
    batch_size=32, 
    output_dir="results/bart/doc2title/vice", 
    lr=8e-4, 
    weight_decay=1e-1, 
    num_epochs=5, 
    chunk_size=1e4, 
    nrows=None, 
    filters=[("publication", "Vice")],
    param_efficient=True
)

# train_title_gen(
#     pretrained_model="google/long-t5-local-base", 
#     batch_size=32, 
#     output_dir="results/long-t5/doc2title/vice", 
#     lr=8e-4, 
#     weight_decay=1e-1, 
#     num_epochs=5, 
#     chunk_size=1e4, 
#     nrows=None, 
#     filters=[("publication", "Vice")],
#     param_efficient=True
# )
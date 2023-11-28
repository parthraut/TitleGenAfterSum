def summarize(args):
    model, max_input_tokens, tokenizer, article = args

    input_ids = tokenizer(article, return_tensors="pt").input_ids
    input_ids = input_ids[:, :max_input_tokens]
    input_ids = input_ids.to(device)

    # Generate output
    output_ids = model.generate(input_ids, max_length=200)

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

def summarize2(args):
    reader, value = args
    return reader(value)

def summarize_csv(dataset_fp):
    print("-----------------------")
    print("Creating All the News 2.0 Summarized dataset")

    print("-----------------------")
    print("Setting model and tokenizer")

    directory_path = f"results/bart/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    model = BartForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    max_input_tokens = model.config.max_position_embeddings
    tokenizer = BartTokenizer.from_pretrained(model_path)

    print("-----------------------")
    print("Reading CSV")
    df = pd.read_csv(dataset_fp, nrows=None)
    df = df.dropna()

    # set_start_method("spawn", force=True)
    # reader = pipeline("summarization", model=model_path, device=device)

    print("-----------------------")
    print("Generating summarization inputs")
    inputs = list()
    for article in tqdm(df["article"], total=len(df["article"])):
        inputs.append((model, max_input_tokens, tokenizer, article))

    # print("-----------------------")
    # print("Generating summarization inputs")
    # inputs = list()
    # for article in tqdm(df["article"], total=len(df["article"])):
    #     inputs.append((reader, article))

    # multi_pool = Pool(processes=num_processors)
    # predictions = multi_pool.map(summarize2, inputs)
    # multi_pool.close()
    # multi_pool.join()
    # print(predictions)

    print("-----------------------")
    print("Summarizing")
    summaries = list()
    for input in tqdm(inputs):
        summary = summarize(input)
        summaries.append(summary)

    print("-----------------------")
    print("Changing dataframe with summaries")
    df["article"] = summaries

    return df

    # return None

def summarize_all():
    print("-----------------------")
    print("Summarizing train")
    train_summ_df = summarize_csv("data/atn-filtered-publication-section-train.csv")
    train_summ_df.to_csv("data/atn-filtered-publication-section-train-bart-summ.csv", index=False)

    print("-----------------------")
    print("Summarizing val")
    val_summ_df = summarize_csv("data/atn-filtered-publication-section-val.csv")
    val_summ_df.to_csv("data/atn-filtered-publication-section-val-bart-summ.csv", index=False)

    print("-----------------------")
    print("Summarizing test")
    test_summ_df = summarize_csv("data/atn-filtered-publication-section-test.csv")
    test_summ_df.to_csv("data/atn-filtered-publication-section-test-bart-summ.csv", index=False)

def train_bart_summ2title(batch_size, save_name, lr_tg, weight_decay, num_epochs, chunk_size=1e4, nrows=None, filters=None, param_efficient=False):
    print("-----------------------")
    print("Training BART for SummTitle")

    directory_path = f"results/bart/doc2summ"
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    model_path = os.path.join(directory_path, subdirectories[-1])

    print("-----------------------")
    print("Setting tokenizer")
    tokenizer = BartTokenizer.from_pretrained(model_path)

    print("-----------------------")
    print("Loading title generation datasets")
    train_ds, val_ds, _ = load_csv_data(
        tokenizer, 
        ("./data/atn-filtered-publication-section-train-summ.csv", "./data/atn-filtered-publication-section-val-summ.csv", "./data/atn-filtered-publication-section-test-summ.csv"),
        chunk_size=chunk_size, 
        nrows=nrows, 
        filters=filters, 
        return_dl=False,
        batch_size=batch_size
    )

    print("-----------------------")
    print("Calling train_bart")
    train_bart(model_path, batch_size, f"results/bart/doc2title_plus/{save_name}", lr_tg, weight_decay, num_epochs, train_ds, val_ds, param_efficient=param_efficient)
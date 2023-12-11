import train_bart
import train_gpt
import sys

print("---------------------")
print("Inside training_slurm")

def main(model_architecture, training_type):
    if model_architecture == "bart":
        print("---------------------")
        print("training_slurm --- BART")

        if training_type == "doc2title":
            train_bart.train_bart_doc2title(
                batch_size=16, 
                save_name="all", 
                lr=8e-4, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=True
            )
        elif training_type == "doc2summ":
            train_bart.train_bart_doc2summ(
                batch_size=16, 
                lr_summ=8e-4, 
                weight_decay=1e-1, 
                num_epochs=2, 
                param_efficient=True
            )
        elif training_type == "doc2title+":
            train_bart.train_bart_doc2title_plus(
                batch_size=16, 
                save_name="all", 
                lr_tg=8e-5, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=True
            )
        elif training_type == "summarize":
            train_bart.summarize_all()
        elif training_type == "summ2title":
            train_bart.train_bart_summ2title(
                batch_size=16, 
                save_name="all", 
                lr_tg=8e-5, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=True
            )

    elif model_architecture == "gpt2":
        print("---------------------")
        print("training_slurm --- GPT2")

        if training_type == "doc2title":
            print("---------------------")
            print("training_slurm --- Doc2Title")

            train_gpt.train_gpt_doc2title(
                batch_size=16, 
                save_name="all", 
                lr=8e-4, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=False
            )
        elif training_type == "doc2summ":
            print("---------------------")
            print("training_slurm --- Doc2Summ")

            train_gpt.train_gpt_doc2summ(
                batch_size=16, 
                lr_summ=8e-4, 
                weight_decay=1e-1, 
                num_epochs=2, 
                param_efficient=False
            )
        elif training_type == "doc2title+":
            print("---------------------")
            print("training_slurm --- Doc2Title+")

            train_gpt.train_gpt_doc2title_plus(
                batch_size=16, 
                save_name="all", 
                lr_tg=8e-5, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=False
            )
        elif training_type == "summarize":
            print("---------------------")
            print("training_slurm --- Summarize")

            train_gpt.summarize_all()
        elif training_type == "summ2title":
            print("---------------------")
            print("training_slurm --- Summ2Title")

            train_gpt.train_gpt_summ2title(
                batch_size=16, 
                save_name="all", 
                lr_tg=8e-5, 
                weight_decay=1e-1, 
                num_epochs=2, 
                chunk_size=1e4, 
                nrows=None, 
                filters=None, 
                param_efficient=False
            )

if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    main(arg1, arg2)
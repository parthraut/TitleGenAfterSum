import pandas as pd
import pdb
import os
import random

def score_data(filename):
    dataset = init_dataset(filename)

    models = ["doc2title", "doc2title+", "summ2title"]

    for i in range(len(dataset)):
        if dataset["doc2title_truthfulness"][i] != -1 and dataset["doc2title_informativeness"][i] != -1 and dataset["doc2title_style"][i] != -1:
            continue

        print("-"*50)
        print(f"Article {i+1} of {len(dataset)}")
        print("-"*50)

        print(dataset["article"][i])

        random.shuffle(models)
        # show all three models title
        print("-"*25 + "TITLE1" + "-"*25)
        print(dataset[models[0]][i])
        print("-"*55)
        print("-"*25 + "TITLE2" + "-"*25)
        print(dataset[models[1]][i])
        print("-"*55)
        print("-"*25 + "TITLE3" + "-"*25)
        print(dataset[models[2]][i])
        print("-"*55)

        continue_flag = True
        while(continue_flag):
            continue_flag = False
            # ask for truthfulness, 3 different numbers 1-10
            truth_scores = input("Truthfullness scores for 3 models? (1-10)")
            truth_scores = truth_scores.split(" ")
            # check each score between 1-10
            if len(truth_scores) != 3:
                print("Invalid number of scores")
                continue_flag = True
                continue

            for score in truth_scores:
                if int(score) < 1 or int(score) > 10:
                    print("Invalid score, must be between 1-10")
                    continue_flag = True
                    break

        # store truthfullness scores
        dataset.at[i, models[0] + "_truthfulness"] = int(truth_scores[0])
        dataset.at[i, models[1] + "_truthfulness"] = int(truth_scores[1])
        dataset.at[i, models[2] + "_truthfulness"] = int(truth_scores[2])

        continue_flag = True
        while(continue_flag):
            continue_flag = False
            # ask for informativeness, 3 different numbers 1-10
            info_scores = input("Informativeness scores for 3 models? (1-10)")
            info_scores = info_scores.split(" ")
            if len(info_scores) != 3:
                print("Invalid number of scores")
                continue_flag = True
                continue
            # check each score between 1-10
            for score in info_scores:
                if int(score) < 1 or int(score) > 10:
                    print("Invalid score")
                    continue_flag = True
                    break
        
        # store informativeness scores
        dataset.at[i, models[0] + "_informativeness"] = int(info_scores[0])
        dataset.at[i, models[1] + "_informativeness"] = int(info_scores[1])
        dataset.at[i, models[2] + "_informativeness"] = int(info_scores[2])

        continue_flag = True
        while(continue_flag):
            continue_flag = False
            # ask for style, 3 different numbers 1-10
            style_scores = input("Style scores for 3 models? (1-10)")
            style_scores = style_scores.split(" ")
            if len(style_scores) != 3:
                print("Invalid number of scores")
                continue_flag = True
                continue
            # check each score between 1-10
            for score in style_scores:
                if int(score) < 1 or int(score) > 10:
                    print("Invalid score")
                    continue_flag = True
                    break


        # store style scores
        dataset.at[i, models[0] + "_style"] = int(style_scores[0])
        dataset.at[i, models[1] + "_style"] = int(style_scores[1])
        dataset.at[i, models[2] + "_style"] = int(style_scores[2])


        # # iterate through each model randomly
        # for model in models:
        #     # print model score name
        #     # print the article
        #     print(dataset["article"][i])
        #     # print the title
        #     print("-"*25 + "TITLE" + "-"*25)
        #     print(dataset[model][i])
        #     print("-"*55)

        #     # ask for truthfulness
        #     if dataset[model + "_truthfulness"][i] == -1:
        #         truthfullness = int(input("Truthfullness score? (1-10)"))
        #         # store truthfullness score
        #         dataset.at[i, model + "_truthfulness"] = truthfullness

        #     # ask for informativeness
        #     if dataset[model + "_informativeness"][i] == -1:
        #         informativeness = int(input("Informativeness score? (1-10)"))
        #         # store informativeness score
        #         dataset.loc[i, model + "_informativeness"] = informativeness

        #     # ask for style
        #     if dataset[model + "_style"][i] == -1:
        #         style = int(input("Style score? (1-10)"))
        #         # store style score
        #         dataset.loc[i, model + "_style"] = style

        #     # print newline
        #     print("\n")
        
        # save dataset
        dataset.to_csv("scored_" + filename, index=False)


# init dataset
def init_dataset(filename):

    scored_dataset_name = "scored_" + filename
 
    if os.path.exists(scored_dataset_name):
        dataset = pd.read_csv(scored_dataset_name)
        return dataset
    
    dataset = pd.read_csv(filename)

    dataset["doc2title_truthfulness"] = -1
    dataset["doc2title+_truthfulness"] = -1
    dataset["summ2title_truthfulness"] = -1

    dataset["doc2title_informativeness"] = -1
    dataset["doc2title+_informativeness"] = -1
    dataset["summ2title_informativeness"] = -1

    dataset["doc2title_style"] = -1
    dataset["doc2title+_style"] = -1
    dataset["summ2title_style"] = -1

    dataset.to_csv(scored_dataset_name, index=False)

    return dataset


def get_averages(filename):
    dataset = pd.read_csv(filename)

    models = ["doc2title", "doc2title+", "summ2title"]

    for model in models:
        print(model)
        # ignore all values of -1 in mean calculation
        print("Truthfullness: ", dataset[model + "_truthfulness"].mean())
        print("Informativeness: ", dataset[model + "_informativeness"].mean())
        print("Style: ", dataset[model + "_style"].mean())
        print("\n")



if __name__ == "__main__":
    # score_data("human_eval_dataset_1024.csv")
    get_averages("scored_human_eval_dataset_1024.csv")
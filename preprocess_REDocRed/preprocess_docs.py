#Original DocRED doesn't have the full paragraphs of documents, so we create them here.

import os
import numpy as np
import json
from tqdm import tqdm
import argparse

def main(doc_split):
    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"

    if doc_split == "train":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\train_revised.json"
        save_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\train_revised_preprocessed.json"
    elif doc_split == "test":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\test_revised.json"
        save_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\test_revised_preprocessed.json"
    elif doc_split == "dev":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\dev_revised.json"
        save_path = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\dev_revised_preprocessed.json"
    else:
        print("Given doc split is not supported!")
        exit()

    with open(data_path) as f:
        data = json.load(f)

    with open(rel_info) as f:
        rels = json.load(f)

    
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(sents_text)
        data[i]["paragraph"] = paragraph

    with open(save_path, 'w',encoding="utf-8") as fout:
        json.dump(data, fout)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--doc_split', type=str, default="train_annotated", help='supported splits {train_distant, train_annotated, dev}')

    # args = parser.parse_args()

    main("train")
    main("test")
    main("dev")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import os
import pandas as pd
from tqdm import tqdm
import argparse


def process_label(raw_dataset, doc_id, label_id, rels, dict_rel_tables):
    #process the corresponding label and get the dataframe out of it.
    paragraph = raw_dataset[doc_id]["paragraph"]
    
    label = raw_dataset[doc_id]["labels"][label_id]
    
    predicate_id = label['r']
    predicate_name = rels[predicate_id]
    
    subject_id = label['h']
    list_dict_subject = raw_dataset[doc_id]["vertexSet"][subject_id]
    subject_names = list(set(map(lambda d: d["name"], list_dict_subject)))
    
    object_id = label['t']
    list_dict_object = raw_dataset[doc_id]["vertexSet"][object_id]
    object_names = list(set(map(lambda d: d["name"], list_dict_object)))
    
    evidences = label["evidence"]
    if dict_rel_tables[predicate_id] is {}:
        ind_ = 0
    else:
        ind_ = len(dict_rel_tables[predicate_id])
    dict_rel_tables[predicate_id][ind_] = [doc_id, paragraph, predicate_name, subject_names, object_names, evidences]
    return dict_rel_tables

def main(doc_split):    

    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"

    if doc_split == "train_distant":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\train_distant_preprocessed.json"
        save_dir = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\relation_docs_distant"
    elif doc_split == "train_annotated":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\train_annotated_preprocessed.json"
        save_dir = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\relation_docs"
    elif doc_split == "dev":
        data_path = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\dev_preprocessed.json"
        save_dir = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\relation_docs_dev"
    else:
        print("Given doc split is not supported!")
        exit()

    with open(data_path) as f:
        data = json.load(f)

    with open(rel_info) as f:
        rels = json.load(f)

    #initialize empty dict of tables

    dict_rel_tables = {}

    for rel in rels.keys():
        dict_rel_tables[rel] = {}

    for i, doc in enumerate(tqdm(data)):
        labels = doc["labels"]
        for j in range(len(labels)):
            dict_rel_tables=process_label(data, i, j, rels, dict_rel_tables)

    for rel in rels.keys():
        dict_rel_tables[rel] = pd.DataFrame.from_dict(dict_rel_tables[rel], orient='index', 
                                                    columns=["paragraph_id", "paragraph", "predicate_name", 
                                                            "subject_names", "object_names", "evidences"])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for k in dict_rel_tables.keys():
        write_path = os.path.join(save_dir, k+".csv")
        dict_rel_tables[k].to_csv(write_path, index=False)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--doc_split', type=str, default="train_annotated", help='supported splits {train_distant, train_annotated, dev}')
    # args = parser.parse_args()
    # main(args)
    main("train_annotated")
    #main("train_distant")
    #main("dev")
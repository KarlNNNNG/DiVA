import json
from tqdm import tqdm
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

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
    # if dict_rel_tables[predicate_id] is {}:
    #     ind_ = 0
    # else:
    #     ind_ = len(dict_rel_tables[predicate_id])
    # dict_rel_tables[predicate_id][ind_] = [doc_id, paragraph, predicate_name, subject_names, object_names, evidences]

    if dict_rel_tables is {}:
        ind_=0
    else:
        ind_ = len(dict_rel_tables)
    dict_rel_tables[ind_] = [doc_id, paragraph, predicate_id, predicate_name, subject_names, object_names, evidences]
    return dict_rel_tables


if __name__ == "__main__":
    data_path =  "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\dev_revised.json"
    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"

    with open(data_path) as f:
        data = json.load(f)

    with open(rel_info) as f:
        rels = json.load(f)
        
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(sents_text)
        data[i]["paragraph"] = paragraph
    
    # dict_rel_tables = {}

    # for rel in rels.keys():
    #     dict_rel_tables[rel] = {}

    # for i, doc in enumerate(tqdm(data)):
    #     labels = doc["labels"]
    #     for j in range(len(labels)):
    #         dict_rel_tables=process_label(data, i, j, rels, dict_rel_tables)

    # for rel in rels.keys():
    #     dict_rel_tables[rel] = pd.DataFrame.from_dict(dict_rel_tables[rel], orient='index', 
    #                                                 columns=["paragraph_id", "paragraph", "predicate_name", 
    #                                                         "subject_names", "object_names", "evidences"])
    

    dict_rel_tables = {}
    for i, doc in enumerate(tqdm(data)):
        labels = doc["labels"]
        for j in range(len(labels)):
            dict_rel_tables = process_label(data, i, j, rels, dict_rel_tables)
    dict_rel_tables = pd.DataFrame.from_dict(dict_rel_tables, orient='index', 
                                            columns=["paragraph_id", "paragraph", "predicate_id", "predicate_name", 
                                            "subject_names", "object_names", "evidences"])

    # for rel in rels.keys():
    #     dict_rel_tables[rel] = pd.DataFrame.from_dict(dict_rel_tables[rel], orient='index', 
    #                                                 columns=["paragraph_id", "paragraph", "predicate_name", 
    #                                                         "subject_names", "object_names", "evidences"])
    write_path  = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\relation_docs_dev.csv"
    dict_rel_tables.to_csv(write_path, index=False)
    # for k in dict_rel_tables.keys():
    #     dict_rel_tables[k].to_csv(write_path, index=False, mode="a")

    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length=512

    print("model is loaded")
    path_doc = "dataset\\redocred\\data\\dev_revised_preprocessed.json"
    with open(path_doc, "rb") as f:
        data = json.load(f)

    list_pars = [doc["paragraph"] for doc in data]
    list_par_ids = [i for i in range(len(data))]
    list_par_titles = [doc["title"] for doc in data]

    embeddings = model.encode(list_pars, show_progress_bar=True, normalize_embeddings=True, batch_size=128)

    #We also keep the length of each doc (i.e. num_words) in case we need a filtering later on 
    num_words = np.array(list(map(lambda x: sum([len(l) for l in x['sents']]), data)))
    write_file = "dataset\\redocred\\data\\embeddings_dev.pkl"

    with open(write_file, 'wb') as fOut:
        pickle.dump({'paragraph_ids':list_par_ids, 'paragraph_titles':list_par_titles, 'embeddings':embeddings, 'num_words':num_words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)



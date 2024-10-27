import json
import pandas as pd
from gpt_api import Demo, LLM
import ast
import pickle
import os
import torch
import re
import numpy as np
import time 
import logging
from tqdm import tqdm
import jsonlines
# [
#   {
#     "instruction": "hi",
#     "input": "",
#     "output": "Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?"
#   },
# ]
def Evaluation(gold, output):
    n_gold = 0
    r_gold = len(gold.keys())
    for key, value in gold.items():
        n_gold += len(value)

    r_pred = len(output.keys())
    n_pred = 0
    for key, value in output.items():
        if value != []:
            n_pred += len(value)
    
    n_correct = 0
    r_correct = 0
    for key, value in output.items():
        if key in gold.keys():
            r_correct += 1
            for i in value:         
                sub = i[0]
                obj = i[1]
                # sub = i[1]
                # obj = i[0]
                for example in gold[key]:
                    #遍历example,examplep[0]里面可能有几个不同名称但指向相同的实体
                    c1 = 0
                    c2 = 0
                    for e in example[0]:
                        if e in sub or sub in e:
                            c1=1
                            break
                    for e in example[1]:
                        if e in obj or obj in e:
                            c2=1
                            break
                    if c1==1 and c2 == 1:
                        n_correct += 1
                        l = gold[key]
                        l.remove(example)
                        gold[key] = l
                        break
        else:
            continue
    return n_pred, n_gold, n_correct, r_pred, r_gold, r_correct

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
    if dict_rel_tables is {}:
        ind_=0
    else:
        ind_ = len(dict_rel_tables)
    dict_rel_tables[ind_] = [doc_id, paragraph, predicate_id, predicate_name, subject_names, object_names, evidences]
    return dict_rel_tables



def CreateRelationAblation(docs, dataset):
    instruct = "Given a passage: { "
    sentence = docs[0][1]
    gold = set()
    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"
    instruct = "Given a passage: { "
    rels = json.load(open(rel_info, "r", encoding="utf-8"))
    rels = '\n'.join(rels.values())
    for i in range(len(docs)):
        gold.add(docs[i][3])
    relation_list = "and a relation_list {" + rels + " }"+"\n"
    format = "Check the passage, and find which relations in the relation list can be derived from the passage. \n \
            Your output format is as following: \n  \
            relation1\n \
            relation2\n\
            relation3\n\
            relation4\n \
            ......"
    prompt_list = instruct+ sentence + " }" +'\n' + relation_list+'\n'+format
    instance = {"instruction":"", 
                "input":"", 
                "output":""}
    instance["instruction"]+= prompt_list
    output = '\n'.join(gold)
    instance["output"]+=output
    dataset.append(instance)
    return dataset 

def CreateRelation(docs, dataset):
    relation_list_dic ={"time":["mouth of the watercourse",
                        "date of birth",
                        "date of death",
                        "publication date",
                        "start time",
                        "end time",
                        "point in time",
                        "narrative location",
                        "work location",], 
                    "organization":["head of government","country",
                                    "country of citizenship",
                                    "continent",
                                    "head of state",
                                    "capital",
                                    "member of sports team",
                                    "member of political party",
                                    "founded by",
                                    "league",
                                    "contains administrative territorial entity",
                                    "headquarters location",
                                    "parent taxon",
                                    "legislative body",
                                    "basin country",
                                    "military branch",
                                    "production company",
                                    "platform",
                                    "residence",
                                    "inception",
                                    "dissolved, abolished or demolished date",
                                    "parent organization",
                                    "product or material produced",
                                    "territory claimed by",
                                    "capital of"],
                    "location":["place of birth",
                            "place of death",
                            "position held",
                            "educated at",
                            #"location",
                            "located in the administrative territorial entity",
                            "headquarters location",
                            "ethnic group",
                            "sister city",
                            "located in or next to body of water",
                            "basin country",
                            "member of",
                            "chairperson",
                            "country of origin",
                            "residence",
                            "located on terrain feature",
                            "location of formation",
                            "territory claimed by",
                            "capital of",],
                    "person":["father",
                            "mother",
                            "spouse",
                            "child",
                            "author",
                            "director",
                            "screenwriter",
                            "composer",
                            "employer",
                            "founded by",
                            "publisher",
                            "owned by",
                            "operator",
                            "cast member", 
                            "producer", 
                            "award received",
                            "creator",
                            "parent taxon",
                            "performer",
                            "manufacturer",
                            "developer",
                            "platform",
                            "lyrics by",
                            "participant",
                            "influenced by",
                            "sibling",],
                    "others":["instance of",
                        "official language",
                        "genre",
                        "religion",
                        "follows",
                        "followed by",
                        "parent taxon",
                        #"series",
                        "record label",
                        "subclass of",
                        "subsidiary",
                        "part of",
                        "original language of work",
                        "platform",
                        "original network",
                        "has part",
                        "conflict",
                        "characters",
                        "notable work",
                        "separated from",
                        "unemployment rate",
                        "participant of",
                        "replaces",
                        "replaced by",
                        "languages spoken, written or signed",
                        "present in work",]}
   
    instruct = "Given a passage: { "
    sentence = docs[0][1]
    gold = set()
    for i in range(len(docs)):
        gold.add(docs[i][3])
    for i in ["time","person","others","location","organization"]:
        relation_list = "and a relation_list {" + "\n ".join(relation_list_dic[i])+ " }"+"\n"
        format = "Check the passage, and find which relations in the relation list can be derived from the passage. \n \
                Your output format is as following: \n  \
                relation1\tTrue\n \
                relation2\tFalse\n\
                relation3\tTrue\n\
                relation4\tFalse\n \
                ......"
        prompt_list = instruct+ sentence + " }" +'\n' + relation_list+'\n'+format
        instance = {"instruction":"", 
                    "input":"", 
                    "output":""}
        instance["instruction"]+= prompt_list
        output = relation_list_dic[i]
        for j in output:
            if j in gold:
                output[output.index(j)] = j+"\tTrue"
            else: 
                output[output.index(j)] = j+"\tFalse"
        instance["output"]+='\n'.join(output)
        dataset.append(instance)
    return dataset 
      
def prepare_template_triplets(docs):
    rels = {}
    for i in range(len(docs)):
        rels[docs[i][3]]=[]
        triplets = {}
    for i in range(len(docs)):
        rels[docs[i][3]].append([docs[i][4], docs[i][5]])
    
    for key, value in rels.items():
        triplet = ""
        for sub, obj in value:
            for k in sub:
                for j in obj:
                    triplet+= "( " + key + " <SEP> " + k + " <SEP> " + j +" )" + "\n"
        triplets[key] = triplet
    # for key, value in rels.items():
    #     triplet = ""
    #     for sub, obj in value:
    #         for k in sub:
    #             for j in obj:
    #                 triplet+= "( " + key + " <SEP> " + j + " <SEP> " + k +" )" + "\n"
    #     triplets[key] = triplet
    return triplets

def CreateTriplets(docs, paragraph_entity, dataset, stage1_rel=None):
    sentence = docs[0][1]
    entity_info = paragraph_entity[sentence]
    if stage1_rel!=None:
        rels = stage1_rel[sentence.replace(" ","")]
        for key in rels:
            instruction = "Given a passage: {} \n Your task is to identify all the unique knowledge triplets of '{}' for the given passage. Knowledge triplet will be ordered as relation, subject entity, and object entity, which are separated by {}. Select the subject entity and object entity from the following entity: \n{}\n \
            The subject entity does the relation to the object entity, object entity is done relation by subject entity. \n \
            If there are multiple triplets, list each of them in a new line. There maybe no triplet present. Follow the example context-relation pairs for the formatting of your output.".format(sentence, key, "[SEP]", entity_info)

            instance = {"instruction":"", 
                        "input":"", 
                        "output":""}
            instance["instruction"]+= instruction
            dataset.append(instance)
        return dataset
    triplets = prepare_template_triplets(docs)
    for key, value in triplets.items():
        # instruction = "Given a passage: {} \n Your task is to identify all the unique knowledge triplets of '{}' for the given passage. Knowledge triplet will be ordered as relation, subject entity, and object entity, which are separated by {}. Select the subject entity and object entity from the following entity: \n{}\n \
        # If there are multiple triplets, list each of them in a new line. There maybe no triplet present. Follow the example context-relation pairs for the formatting of your output.".format(sentence, key, "[SEP]", entity_info)
        instruction = "Given a passage: {} \n Your task is to identify all the unique knowledge triplets of '{}' for the given passage. Knowledge triplet will be ordered as relation, subject entity, and object entity, which are separated by {}. Select the subject entity and object entity from the following entity: \n{}\n \
                The subject entity does the relation to the object entity, object entity is done relation by subject entity. \n \
                If there are multiple triplets, list each of them in a new line. There maybe no triplet present. Follow the example context-relation pairs for the formatting of your output.".format(sentence, key, "[SEP]", entity_info)

        instance = {"instruction":"", 
                    "input":"", 
                    "output":""}
        instance["instruction"]+= instruction
        instance["output"]+= value
        dataset.append(instance)
    return dataset


def processStage1(data_path, stage1_input_file, stage1_output_file, stage2_input_file, stage2_output_file, paragraph_entity_file):
    llm = Demo()

    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"
    with open(rel_info) as f:
        rel_list = json.load(f)
    # with open(stage1_input_file) as f:
    #     stage1_input = json.load(f)
    


    # stage1_output = stage1_input
    # for i in range(len(stage1_input)):
    #     text = stage1_input[i]["instruction"]
    #     rels = []
    #     output = llm.get_multiple_sample([text])
    #     output = output.split("\n")
    
    #     for rel in output:
    #         if "True" in rel and rel not in rels:
    #             rels.append(rel.split("\t")[0])
    #     temp = []
    #     for rel in rels:
    #         if rel ==" ":
    #             continue 
    #         if rel == "languages spoken, written or signed" or "dissolved, abolished or demolished":
    #             temp.append(rel)
    #             continue
    #         rel = re.findall("[a-zA-Z ]+", rel)
    #         if rel ==[]:
    #             continue
    #         rel = rel[0].strip(" ")
    #         if rel not in rel_list.values():
    #             continue
    #         temp.append(rel)
    #     stage1_output[i]["output"] = "\n".join(temp)
    # json_str = json.dumps(stage1_output, indent=2)
    # with open(stage1_output_file, 'w') as json_file:
    #     json_file.write(json_str)

    # stage1_relation = {}
    # with open(data_path) as f:
    #     data = json.load(f)

    # for i in tqdm(range(len(data))):
    #     sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
    #     data[i]["sents_text"] = sents_text
        
    #     paragraph = " ".join(sents_text)
    #     data[i]["paragraph"] = paragraph

    
    # for i in tqdm(range(len(stage1_output))):
    #     rels = stage1_output[i]["output"].split("\n")
    #     prompt = stage1_output[i]["instruction"]

    #     sentence = re.findall("Given a passage: {.+}", prompt)[-1]

    #     sentence = sentence[18:-2]
    #     sentence = sentence.replace(" ", "")
    #     if sentence not in stage1_relation.keys():
    #         stage1_relation[sentence] = rels
    #     else:
    #         stage1_relation[sentence] = stage1_relation[sentence]+rels
    
    

    # dict_rel_tables = {}
    # for i, doc in enumerate(tqdm(data)):
    #     labels = doc["labels"]
    #     for j in range(len(labels)):
    #         dict_rel_tables = process_label(data, i, j, rel_list, dict_rel_tables)

    # paragraph_entity = torch.load(paragraph_entity_file)
    # docs = []
    # stage2_input = []
    # for i in tqdm(range(len(dict_rel_tables))):
    #     if docs == []:
    #         docs.append(dict_rel_tables[i])
    #     elif docs[-1][0] == dict_rel_tables[i][0]:
    #         docs.append(dict_rel_tables[i])
    #     else:
    #         stage2_input = CreateTriplets(docs, paragraph_entity, stage2_input, stage1_relation)
    #         docs = [dict_rel_tables[i]]
    # if docs != []:
    #     stage2_input =  CreateTriplets(docs, paragraph_entity, stage2_input, stage1_relation)
    # json_str = json.dumps(stage2_input, indent=2)
    # with open(stage2_input_file, 'w') as json_file:
    #     json_file.write(json_str)

    #Stage2 begin
    stage2_input = []
    with open(stage2_input_file, 'r', encoding="utf-8") as f:
        stage2_input = json.load(f)

    stage2_output = stage2_input
    for i in range(len(stage2_input)):
        text = stage2_input[i]["instruction"]
        results = llm.get_multiple_sample([text])
        stage2_output[i]["output"] = results

    stage1_relation = {}
    with open(data_path) as f:
        data = json.load(f)

    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"
    with open(rel_info) as f:
        rels = json.load(f)
    
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(sents_text)
        data[i]["paragraph"] = paragraph
   
    dict_rel_tables = {}
    for i, doc in enumerate(tqdm(data)):
        labels = doc["labels"]
        for j in range(len(labels)):
            dict_rel_tables = process_label(data, i, j, rels, dict_rel_tables)
    
    sentence_triplets_gold = {}
    docs = []
    for i in tqdm(range(len(dict_rel_tables))):
        if docs == []:
            docs.append(dict_rel_tables[i])
        elif docs[-1][0] == dict_rel_tables[i][0]:
            docs.append(dict_rel_tables[i])
        else:
            #process gold triplets
            sentence = docs[0][1].replace(" ", "")
            gold = {}

            for j in range(len(docs)):
                if docs[j][3] not in gold.keys():
                    gold[docs[j][3]] = [[docs[j][4], docs[j][5]]]
                else:
                    gold[docs[j][3]].append([docs[j][4], docs[j][5]])
            sentence_triplets_gold[sentence] = gold
            # store new docs
            docs = [dict_rel_tables[i]]
    if docs != []:
        for j in range(len(docs)):
            if docs[j][3] not in gold.keys():
                gold[docs[j][3]] = [[docs[j][4], docs[j][5]]]
            else:
                gold[docs[j][3]].append([docs[j][4], docs[j][5]])
        sentence_triplets_gold[sentence] = gold
    #process output triplets
    sentence_triplets = {}
    for i in tqdm(range(len(stage2_output))):
        sentence = stage2_output[i]["instruction"].split("\n")[2][16:-1]
        sentence = sentence.replace(" ", "")
        raw_triplets = stage2_output[i]["output"].split("\n")
        triplets ={}
        for line in raw_triplets:
            if "<SEP>" not in line or len(line.split("[SEP]")) < 3:
                print("Attention")
                continue
            rel = line.split("[SEP]")[0].strip(" ")
            sub = line.split("[SEP]")[1].strip(" ")
            obj = line.split("[SEP]")[2].strip(" ").strip("\r").strip(")")
            if rel not in triplets.keys():
                triplets[rel] = [[sub, obj]]
            else:
                triplets[rel].append([sub, obj])
        if sentence in sentence_triplets.keys():
            sentence_triplets[sentence] = dict(sentence_triplets[sentence], **triplets)  
        else:
            sentence_triplets[sentence] = triplets
    n_pred, n_gold, n_correct = 0,0,0
    r_pred, r_gold, r_correct = 0,0,0
    for key, value in sentence_triplets_gold.items():
        gold = value
        if key not in sentence_triplets.keys():
            print(key)
            nn_gold = 0
            rr_gold = len(gold.keys())
            for k, v in gold.items():
                nn_gold += len(value)
            n1,n2,n3, n4, n5, n6 = 0, nn_gold, 0, rr_gold, 0 , 0
        else:
            pred = sentence_triplets[key]
            n1,n2,n3, n4, n5, n6 = Evaluation(gold, pred)
        n_pred += n1 
        n_gold += n2
        n_correct += n3
        r_pred += n4
        r_gold += n5 
        r_correct += n6
    print("n_pred {}. n_gold {}. n_correct {}".format(n_pred, n_gold, n_correct))
    print(compute_f1(n_pred, n_gold, n_correct))    
    print("r_pred {}. r_gold {}. r_correct {}".format(r_pred, r_gold, r_correct))
    print(compute_f1(r_pred, r_gold, r_correct))

def compute_f1(n_pred, n_gold, n_correct):
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 /n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 =0.0
        return {"precision": prec, "recall": recall, "f1": f1}
    

if __name__ == "__main__":   

    data_path = r"dataset/docred/dev.json"
    stage1_input_file = r"D:\Projects\GenerativeRE\dataset\docred\data\sp_Stage12\dev_Stage1.json"
    stage1_output_file = r"dataset/docred/data/sp_Stage12/zeroshot/dev_stage1_output.json"
    stage2_input_file = r"dataset/docred/data/sp_Stage12/zeroshot/dev_stage2_input.json"
    stage2_output_file = r"dataset/docred/data/sp_Stage12/zeroshot/dev_stage2_output.json"
    paragraph_entity_file =  r"D:\Projects\GenerativeRE\dataset\docred\data\sp_Stage12\paragraph_entityDev.pt"
    processStage1(data_path, stage1_input_file, stage1_output_file, stage2_input_file, stage2_output_file, paragraph_entity_file)
    raw_stage2_output = []
    file = ""
    with open(file, encoding="utf-8") as f:
        for line in jsonlines.Reader(f):
            raw_stage2_output.append(line)
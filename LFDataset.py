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
                                    "dissolved, abolished or demolished",
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

def LFDataset(type):
    #Stage 1 - Extraction Relation
    data_path =  r"D:\Projects\GenerativeRE\dataset\docred\data\train_annotated.json"
    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"
    instruct = "Given a passage: { "
    with open(data_path) as f:
        data = json.load(f)

    with open(rel_info) as f:
        rels = json.load(f)

    paragraph_entity = {}    
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(sents_text)
        data[i]["paragraph"] = paragraph
    
    dict_rel_tables = {}

    for i, doc in enumerate(tqdm(data)):
        labels = doc["labels"]
        entity = {"BLANK":[], "ORG": [], "LOC": [], "TIME": [], "PER": [], "MISC": [], "NUM": []}
        paragraph = data[i]["paragraph"]
        # use groundtruth entity info
        # for j in data[i]["vertexSet"]:
        #     for k in j:
        #         entity[k["type"]].append(k["name"])
        # entity_info = ""
        # for key in entity.keys():
        #     entity_info += str(key)+":\n"
        #     entity_info += "\n".join(entity[key])
        #use chatgpt entity info
    #     entity_info = ""
    #     format_entity = "Find out all entities whose tags belong to: \n \
    # ORG (Organization): Represents organizations, such as companies, institutions, government departments, etc.\n \
    # LOC (Location): Represents geographical locations, such as countries, cities, rivers, country of citizenship, etc.\n \
    # TIME (Time): Represents time-related information, such as dates, time points, time periods, etc.\n \
    # PER (Person): Represents names of people.\
    # MISC (Miscellaneous): Represents other categories of entities, usually those that do not fall into any of the above types.\n \
    # NUM (Number): Represents numbers or quantities, such as years, amounts of money, percentages, etc. \n \
    # Your output format is as following:\n \
    # ORG:\n \
    # entity1\n \
    # LOC:\n \
    # entity2\n \
    # TIME:\n \
    # entity3\n \
    # PER:\n \
    # entity4\n \
    # MISC:\n \
    # entity5\n \
    # NUM:\n \
    # entity6\n "
    #     prompt_list_entity = [instruct+ paragraph + " }" +'\n' +format_entity]
    #     entity_info = model.get_multiple_sample(prompt_list_entity)

    #     paragraph_entity[paragraph] = entity_info
        for j in range(len(labels)):
            dict_rel_tables = process_label(data, i, j, rels, dict_rel_tables)
    #torch.save(paragraph_entity, r"dataset\redocred\data\sub-predicate\paragraph_entityTest.pt")
    paragraph_entity = torch.load(r"D:\Projects\GenerativeRE\dataset\docred\data\sp_Stage12\paragraph_entityTrain.pt")
    docs = []
    dataset = []
    for i in tqdm(range(len(dict_rel_tables))):
        if docs == []:
            docs.append(dict_rel_tables[i])
        elif docs[-1][0] == dict_rel_tables[i][0]:
            docs.append(dict_rel_tables[i])
        else:
            dataset = CreateRelationAblation(docs, dataset)
            docs = [dict_rel_tables[i]]
    if docs != []:
        dataset = CreateRelationAblation(docs, dataset)
    json_str = json.dumps(dataset, indent=2)
    with open(r"D:\Projects\GenerativeRE\dataset\docred\data\ablation\Stage1\train_Stage1.json", 'w') as json_file:
        json_file.write(json_str)
    
    #Stage 2 - Extraction Entity
    docs = []
    dataset = []
    for i in tqdm(range(len(dict_rel_tables))):
        if docs == []:
            docs.append(dict_rel_tables[i])
        elif docs[-1][0] == dict_rel_tables[i][0]:
            docs.append(dict_rel_tables[i])
        else:
            dataset = CreateTriplets(docs, paragraph_entity, dataset)
            docs = [dict_rel_tables[i]]
    if docs != []:
        dataset =  CreateTriplets(docs, paragraph_entity, dataset)
    json_str = json.dumps(dataset, indent=2)
    with open(r"D:\Projects\GenerativeRE\dataset\docred\data\ablation\Stage1\train_Stage2.json", 'w') as json_file:
        json_file.write(json_str)

def processStage1(data_path, file):

    raw_stage1_output = []
    with open(file, encoding="utf-8") as f:
        for line in jsonlines.Reader(f):
            raw_stage1_output.append(line)

    stage1_relation = {}
    with open(data_path) as f:
        data = json.load(f)

    rel_info = "D:\\Projects\\GenerativeRE\\dataset\\docred\\data\\rel_info.json"
    
    
    for i in tqdm(range(len(data))):
        sents_text = list(map(lambda words: " ".join(words), data[i]["sents"]))
        data[i]["sents_text"] = sents_text
        
        paragraph = " ".join(sents_text)
        if "The Palestinian National Theatre" in paragraph:
            print("........")
        data[i]["paragraph"] = paragraph



    for i in tqdm(range(len(raw_stage1_output))):
        raw_rels = raw_stage1_output[i]["predict"].split("\n")
        # ablation stage1
        #rels = raw_rels

        rels = []
        for j in raw_rels:
            if "True" in j:
                rels.append(j.split("\t")[0])
        prompt = raw_stage1_output[i]["prompt"]

        sentence = re.findall("Given a passage: {.+}", prompt)[-1]

        #sentence = sentence.strip("Given a passage: {")[18:-2]
        #sentence = sentence[19:-2]
        sentence = sentence[18:-2]
        sentence = sentence.replace(" ", "")
        if sentence not in stage1_relation.keys():
            stage1_relation[sentence] = rels
        else:
            stage1_relation[sentence] = stage1_relation[sentence]+rels
    
    
    with open(rel_info) as f:
        rels = json.load(f)
    dict_rel_tables = {}
    for i, doc in enumerate(tqdm(data)):
        labels = doc["labels"]
        for j in range(len(labels)):
            dict_rel_tables = process_label(data, i, j, rels, dict_rel_tables)

    paragraph_entity = torch.load(r"D:\Projects\GenerativeRE\dataset\redocred\data\sub-predicate\paragraph_entityTest.pt")
    docs = []
    dataset = []
    for i in tqdm(range(len(dict_rel_tables))):

        if docs == []:
            docs.append(dict_rel_tables[i])
        elif docs[-1][0] == dict_rel_tables[i][0]:
            docs.append(dict_rel_tables[i])
        else:
            dataset = CreateTriplets(docs, paragraph_entity, dataset, stage1_relation)
            docs = [dict_rel_tables[i]]
    if docs != []:
        dataset =  CreateTriplets(docs, paragraph_entity, dataset, stage1_relation)
    json_str = json.dumps(dataset, indent=2)
    with open(r"D:\Projects\GenerativeRE\dataset\redocred\data\ablation\Stage2\test_Stage2_input.json", 'w') as json_file:
        json_file.write(json_str)

def evalStage2(data_path, file):
    raw_stage2_output = []
    with open(file, encoding="utf-8") as f:
        for line in jsonlines.Reader(f):
            raw_stage2_output.append(line)

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
    for i in tqdm(range(len(raw_stage2_output))):
        sentence = raw_stage2_output[i]["prompt"].split("\n")[2][16:-1]
        sentence = sentence.replace(" ", "")
        raw_triplets = raw_stage2_output[i]["predict"].split("\n")
        triplets ={}
        for line in raw_triplets:
            if "<SEP>" not in line or len(line.split("<SEP>")) < 3:
                continue
            rel = line.split("<SEP>")[0][2:-1]
            sub = line.split("<SEP>")[1][1:-1]
            obj = line.split("<SEP>")[2][1:-2]
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
    #LFDataset("redocred test")
    #processStage1(r"D:\Projects\GenerativeRE\dataset\redocred\data\test_revised.json", r"D:\Projects\GenerativeRE\dataset\redocred\data\ablation\Stage2\test_Stage1_output.jsonl")
    gold_file = r"D:\Projects\GenerativeRE\dataset\redocred\data\test_revised.json"
    stage2file = r"dataset/redocred/data/sub-predicate/test_Stage2_output.jsonl"
    evalStage2(gold_file, stage2file)


# ABLATION docred stage1
#     n_pred 15906. n_gold 12275. n_correct 6420
# {'precision': 0.4036212749905696, 'recall': 0.5230142566191446, 'f1': 0.45562613108122496}
# r_pred 4979. r_gold 5255. r_correct 3590
# {'precision': 0.7210283189395461, 'recall': 0.6831588962892483, 'f1': 0.7015829587649013}

# ABLATION redocred stage1 dev
# n_pred 17916. n_gold 17284. n_correct 10255
# {'precision': 0.572393391382005, 'recall': 0.5933233047905577, 'f1': 0.5826704545454545}
# r_pred 4120. r_gold 4989. r_correct 3619
# {'precision': 0.8783980582524272, 'recall': 0.7253958709160152, 'f1': 0.7945987484905039}

# ABLATION redocred stage1 test
# n_pred 17067. n_gold 17448. n_correct 9760
# {'precision': 0.5718638307845549, 'recall': 0.5593764328289775, 'f1': 0.5655512096190063}
# r_pred 4022. r_gold 4766. r_correct 3476
# {'precision': 0.8642466434609647, 'recall': 0.7293327738145196, 'f1': 0.7910787437414656}




# ABLATION docred stage2
# n_pred 16206. n_gold 12299. n_correct 6684
# {'precision': 0.41243983709737136, 'recall': 0.5434588177900642, 'f1': 0.4689703560778811}
# r_pred 4813. r_gold 5242. r_correct 3693
# {'precision': 0.7672969042177437, 'recall': 0.7045020984357115, 'f1': 0.7345599204375933}

# ABLATION redocred stage2 dev
# n_pred 14040. n_gold 17284. n_correct 8858
# {'precision': 0.6309116809116809, 'recall': 0.5124971071511224, 'f1': 0.5655727237900651}
# r_pred 3659. r_gold 4989. r_correct 3365
# {'precision': 0.9196501776441651, 'recall': 0.6744838645019042, 'f1': 0.7782146160962072}

# ABLATION redocred stage2 test
# n_pred 13239. n_gold 17448. n_correct 8457
# {'precision': 0.6387944708814866, 'recall': 0.484697386519945, 'f1': 0.5511780232671815}
# r_pred 3502. r_gold 4766. r_correct 3242
# {'precision': 0.9257567104511708, 'recall': 0.6802349979018044, 'f1': 0.7842283502660861}
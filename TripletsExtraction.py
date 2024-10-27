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

rel_list = json.load(open(r"D:\Projects\GenerativeRE\dataset\docred\rel_info.json"))
rel_list = {rel_list[name]:name for name in rel_list.keys()}

def get_logger(name, store_path):
    logger = logging.getLogger(name)
    # 创建一个handler，用于写入日志文件

    fh = logging.FileHandler(store_path, mode='w+', encoding='utf-8')
    # 再创建一个handler用于输出到控制台
    ch = logging.StreamHandler()
    # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    # 定义日志输出层级
    logger.setLevel(logging.DEBUG)
    # 定义控制台输出层级
    # logger.setLevel(logging.DEBUG)
    # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
    fh.setFormatter(formatter)
    # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
    ch.setFormatter(formatter)
    # 给logger对象绑定文件操作符
    logger.addHandler(fh)
    # 给logger对象绑定文件操作符
    logger.addHandler(ch)
    return logger

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
    
#BELOW is triplets for (rel, sub, obj)
def construct_triplet(rel_name, subject_name, object_name, ):
    triplet = "(" + rel_name + "[SEP]"
    if subject_name is not None:
        triplet += " " + subject_name + "[SEP]"

        if object_name is not None:
            triplet += " " + object_name + ")"

    return triplet

def prepare_template(data, template_indices, is_eval=False, rel=None):
    template = ""
    doc_prefix="Context: "
    relation_prefix = "Relation: "

    for ind in template_indices[0]:
        paragraph = data.loc[ind].paragraph
        rel_name = data.loc[ind].predicate_name
        subject_names_list = data.loc[ind].subject_names
        object_names_list = data.loc[ind].object_names



        if is_eval:
            template += doc_prefix
            template += paragraph + "\n"
            #template += self.args.relation_prefix[:-1] This was for prefix

            template += relation_prefix
            template += "("+rel#This is for providing rel name.

            #template += construct_triplet(rel_name, None, None, self.args)
            #pass
        else:
            template += "Example: "
            template += paragraph + "\n"
            for i in range(len(subject_names_list)):
                template += relation_prefix
                template += construct_triplet(rel_name[i], subject_names_list[i][0], object_names_list[i][0])
                if i == len(subject_names_list)-1:
                    template += "\n\n"
                else:
                    template += "\n"

    return template

def get_template_indices(data, index, context):
    #Returns the random n examples (for each doc) out of closest topk examples to generate the context

    #Initialize randomizer with a seed for reproducibilty
    np.random.seed(666)

    topk = 3
    # 虽然在dataframe中有index这个列名 但在此处index实际上为paragraph_id 
    data_list = data[data['paragraph_id']==index].values.tolist()
    emb_data = np.array(data_list[0][-2])
    emb_context = np.array(context.embeddings.tolist())

    similarities = emb_data.dot(emb_context.T)
    #If we are using same docs for both data and context, we don't allow test doc to appear in context
    # if path_data == path_context:
    #     np.fill_diagonal(similarities, -1)

    #Get top indices for each row (i.e. test instance)
    #template_indices_topk = np.argsort(-similarities)[:topk]
    sort_similarities = np.argsort(-similarities)
    template_indices_topk = [sort_similarities[:topk]]

    #Now pick random num_examples examples out of topk examples
    #template_indices = np.array([np.random.choice(row, num_examples, replace=False) for row in template_indices_topk])

    #For the templace indices, we collect their similarity scores
    #sims_template_indices = np.array([similarities[s, indices] for s, indices in zip(np.arange(len(emb_data)), template_indices)])
    
    sims_template_indices = [list(similarities[template_indices_topk])]

    return template_indices_topk, sims_template_indices

def EntityExtraction(data, index, key, log, entity):
    context = torch.load(os.path.join(r"D:\Projects\GenerativeRE\dataset\redocred\data\relation_docs_train", rel_list[key]+".pt"))
    # 虽然在dataframe中有index这个列名 但在此处index实际上为paragraph_id 而得到的结果template_indices反而是真正的index
    template_indices, similarities_template_indices = get_template_indices(data, index, context)
    # instruction = "Your task is to identify all the unique knowledge triplets of '{}' for a given context. Knowledge triplet will be ordered as relation, subject, and object, which are separated by {}. If there are multiple triplets, list each of them in a new line. Follow the example context-relation pairs for the formatting of your output.".format(key, "[SEP]")
    instruction = "Your task is to identify all the unique knowledge triplets of '{}' for a given context. Knowledge triplet will be ordered as relation, subject entity, and object entity, which are separated by {}. Select the subject entity and object entity from the following entity: \n{}\n \
    If there are multiple triplets, list each of them in a new line. There maybe no triplet present. Follow the example context-relation pairs for the formatting of your output.".format(key, "[SEP]", entity)
    # There must be at least one triplet present
    # 因为index实际上是paragraph_id所以需要转一下
    ind = data[data['paragraph_id']==index].index.values.tolist()
    # prepare_template的输入要求是真正的index
    prompt = instruction + "\n\n" + prepare_template(context, template_indices) + prepare_template(data, [ind], is_eval=True, rel=key)
    log.info(prompt)
    log.info("-"*30)
    entitypair = model.get_multiple_sample([prompt])
    return entitypair

def RelationExtraction(model, sentence, log):
    relation_list_dic ={"time":["mouth of the watercourse",
                        "date of birth",
                        "date of death",
                        "publication date",
                        "start time",
                        "end time",
                        "point in time",
                        #"narrative location",
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
                        "series",
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
    relation_desc = json.load(open(r"D:\Projects\GenerativeRE\dataset\redocred\data\relation_description_redocred.json"))
    for key, value in relation_list_dic.items():
        temp = []
        for i in range(len(value)):
            temp.append(value[i] + ": ") #+ relation_desc[value[i]])
        relation_list_dic[key] = temp

    #relation_list = "and a relation_list \'head of government\', \'country\', \'place of birth\', \'place of death\', \'father\', \'mother\', \'spouse\', \'country of citizenship\', \'continent\', \'instance of\', \'head of state\', \'capital\', \'official language\', \'position held\', \'child\', \'author\', \'member of sports team\', \'director\', \'screenwriter\', \'educated at\', \'composer\', \'member of political party\', \'employer\', \'founded by\', \'league\', \'publisher\', \'owned by\', \'located in the administrative territorial entity\', \'genre\', \'operator\', \'religion\', \'contains administrative territorial entity\', \'follows\', \'followed by\', \'headquarters location\', \'cast member\', \'producer\', \'award received\', \'creator\', \'parent taxon\', \'ethnic group\', \'performer\', \'manufacturer\', \'developer\', \'series\', \'sister city\', \'legislative body\', \'basin country\', \'located in or next to body of water\', \'military branch\', \'record label\', \'production company\', \'location\', \'subclass of\', \'subsidiary\', \'part of\', \'original language of work\', \'platform\', \'mouth of the watercourse\', \'original network\', \'member of\', \'chairperson\', \'country of origin\', \'has part\', \'residence\', \'date of birth\', \'date of death\', \'inception\', \'dissolved, abolished or demolished\', \'publication date\', \'start time\', \'end time\', \'point in time\', \'conflict\', \'characters\', \'lyrics by\', \'located on terrain feature\', \'participant\', \'influenced by\', \'location of formation\', \'parent organization\', \'notable work\', \'separated from\', \'narrative location\', \'work location\', \'applies to jurisdiction\', \'product or material produced\', \'unemployment rate\', \'territory claimed by\', \'participant of\', \'replaces\', \'replaced by\', \'capital of\', \'languages spoken, written or signed\', \'present in work\', \'sibling\'"

    instruct = "Given a passage: { "

    # format = "Check the passage, and find which relations can be derived from the passage. \n \
    #         Your output format is as following: \n \
    #         relation1 \n \
    #         relation2 \n \
    #         ...... \n  \
    #         one example like: \n \
    #         country of citizenship \n \
    #         father \n \
    #         ...... \n \
    #         The relations must be in the relation list.  \n "

    format_entity = "Find out all entities whose tags belong to: \n \
                    ORG (Organization): Represents organizations, such as companies, institutions, government departments, etc.\n \
                    LOC (Location): Represents geographical locations, such as countries, cities, rivers, country of citizenship, etc.\n \
                    TIME (Time): Represents time-related information, such as dates, time points, time periods, etc.\n \
                    PER (Person): Represents names of people.\
                    MISC (Miscellaneous): Represents other categories of entities, usually those that do not fall into any of the above types.\n \
                    NUM (Number): Represents numbers or quantities, such as years, amounts of money, percentages, etc.\
                    Your output format is as following:\n \
                    ORG:\n \
                    entity1\n \
                    LOC:\n \
                    entity2\n \
                    TIME:\n \
                    entity3\n \
                    PER:\n \
                    entity4\n \
                    MISC:\n \
                    entity5\n \
                    NUM:\n \
                    entity6\n "
    prompt_list_entity = [instruct+ sentence + " }" +'\n' +format_entity]
    entity = model.get_multiple_sample(prompt_list_entity)
    rels = []

    for i in ["time","person","others","location","organization"]:
        relation_list = "and a relation_list {" + "\n ".join(relation_list_dic[i])+ " }"+"\n"
        format = "Check the passage, and find which relations in the relation list can be derived from the passage. \n \
                Your output format is as following: \n  \
                relation1\tTrue\n \
                relation2\tFalse\n\
                relation3\tTrue\n\
                relation4\tFalse\n \
                ......"
        prompt_list = [instruct+ sentence + " }" +'\n' + relation_list+'\n'+format]
        output = model.get_multiple_sample(prompt_list)
        output = output.split("\n")
    
        for rel in output:
            if "True" in rel and rel not in rels:
                rels.append(rel.split("\t")[0])
    log.info(prompt_list)
    log.info(rels)
    log.info("-"*30)

    return rels, entity

def Evaluation(gold, output):
    n_gold = 0
    r_gold = len(gold.keys())
    for key, value in gold.items():
        n_gold += len(value)

    n_pred = 0
    r_pred = len(output.keys())
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
                    # if sub in example[0] and obj in example[1]:
                    #     n_correct += 1
                    #     l = gold[key]
                    #     l.remove(example)
                    #     gold[key] = l
                    #     break
                    # elif example[0] in sub and example[1] in obj:
                    #     n_correct += 1
                    #     l = gold[key]
                    #     l.remove(example)
                    #     gold[key] = l
                    #     break
        else:
            continue

    return n_pred, n_gold, n_correct, r_pred, r_gold, r_correct

if __name__ == "__main__":       
    # Obtain development embeddings and all information
 
    raw_data = pd.read_csv(r"D:\Projects\GenerativeRE\dataset\redocred\data\relation_docs_dev.csv")
    raw_data.subject_names = raw_data.subject_names.apply(ast.literal_eval)
    raw_data.object_names = raw_data.object_names.apply(ast.literal_eval)
    raw_data.evidences = raw_data.evidences.apply(ast.literal_eval)

    agg_dict={"paragraph": lambda x: list(x)[0],
        #"len_paragraph": lambda x: list(x)[0],
        "predicate_name":lambda x: list(x),
        "subject_names":lambda x: list(x),
        "object_names":lambda x: list(x),
        "evidences":lambda x: list(x)}

    raw_data = raw_data.groupby("paragraph_id").agg(agg_dict)
    raw_data = raw_data.reset_index()

    # We also keep track of how many relations (for our predicate) exists, which will be used to filter out certain num_rels in context
    raw_data["num_rels"] = raw_data.subject_names.apply(len)
    path_embeddings = r"D:\Projects\GenerativeRE\dataset\redocred\data\embeddings_dev.pkl"
    with open(path_embeddings, "rb") as fIn:
        emb_data = pickle.load(fIn)
        emb_data["embeddings"] = emb_data["embeddings"].tolist()
        #simple name change for a key
        emb_data['paragraph_id'] = emb_data.pop('paragraph_ids')
        emb_data = pd.DataFrame.from_dict(emb_data) 
        emb_data = emb_data.set_index('paragraph_id')
        raw_data = raw_data.merge(emb_data, how='inner', on='paragraph_id')
        raw_data = raw_data.reset_index()
    raw_data = raw_data.drop('index', axis=1)
    data = raw_data

    # Obtain Retrieval embeddings and all information devided by Relation TYPE

    # for item in os.scandir("D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\relation_docs_train"):
    #     if ".pt" in item.path:
    #         continue
    #     raw_data = pd.read_csv(item.path)
    #     raw_data.subject_names = raw_data.subject_names.apply(ast.literal_eval)
    #     raw_data.object_names = raw_data.object_names.apply(ast.literal_eval)
    #     raw_data.evidences = raw_data.evidences.apply(ast.literal_eval)
    #     agg_dict={"paragraph": lambda x: list(x)[0],
    #              #"len_paragraph": lambda x: list(x)[0],
    #              "predicate_name":lambda x: list(x),
    #              "subject_names":lambda x: list(x),
    #              "object_names":lambda x: list(x),
    #              "evidences":lambda x: list(x)}

    #     raw_data = raw_data.groupby("paragraph_id").agg(agg_dict)
    #     raw_data = raw_data.reset_index()

    #     #We also keep track of how many relations (for our predicate) exists, which will be used to filter out certain num_rels in context
    #     raw_data["num_rels"] = raw_data.subject_names.apply(len)

    #     path_embeddings = "D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\embeddings_train.pkl"
    #     with open (path_embeddings, "rb") as fIn:
    #         emb_data = pickle.load(fIn)
    #         emb_data["embeddings"] = emb_data["embeddings"].tolist()
    #         #simple name change for a key
    #         emb_data['paragraph_id'] = emb_data.pop('paragraph_ids')
    #         emb_data = pd.DataFrame.from_dict(emb_data) 
    #         emb_data = emb_data.set_index('paragraph_id')
    #         raw_data = raw_data.merge(emb_data, how='inner', on='paragraph_id')
    #         raw_data = raw_data.reset_index()
    #     torch.save(raw_data, item.path.strip(".csv")+".pt" )
        
    # logger
    filetime = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    store_path = ".\\results\\knn_{}_{}".format("end2end", filetime)
    
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    log = get_logger('result', store_path+"/result.log")
    model = Demo()
    n_pred, n_gold, n_correct = 0, 0, 0
    r_pred, r_gold, r_correct = 0, 0, 0
    all_output = {}
    for index in range(998):
        all_output[index] = []
        if data[data['paragraph_id']==index].size==0:
            continue

        data_list = data[data['paragraph_id']==index].values.tolist()
        sentence = data_list[0][1]
        embeddings = torch.tensor(data_list[0][-2])
        gold = {}
        assert len(data_list[0][2]) == len(data_list[0][3]) == len(data_list[0][4])
        for i in range(len(data_list[0][2])):
            gold[data_list[0][2][i]] = []
        for i in range(len(data_list[0][2])):
            gold[data_list[0][2][i]].append([data_list[0][3][i], data_list[0][4][i]])

        rels, entity= RelationExtraction(model, sentence, log)
        #rels = rels.split("\n")
        output = {}
        for rel in rels:
            if rel ==" ":
                continue    
            #rel = rel.split("reason:")[0].strip(" ")

            rel = re.findall("[a-zA-Z ]+", rel)
            if rel ==[]:
                continue
            rel = rel[0].strip(" ")
            if rel not in rel_list.keys():
                continue
            output[rel] = []

        for rel in output.keys():
            triplets= EntityExtraction(data, index, rel, log, entity)
            log.info(triplets)
                #triplets = "(country[SEP] Piraeus[SEP] Greece)\nRelation: (country[SEP] Athens[SEP] Greece)\nRelation: (country[SEP] Digea[SEP] Greece)"    
                # transform the triplets from generated GPT
            triplets = triplets.split("\n")

            triplets = list(filter(None, triplets))
            for i in range(len(triplets)):
                triplet = triplets[i].split("[SEP]")
                triplet = list(filter(lambda x: x !="", triplet))
                if len(triplet) == 3:
                    sub = triplet[1].strip(" ")
                    obj = triplet[2].strip(" ").strip("\r").strip(")")
                    output[rel].append([sub, obj])
                elif len(triplet) == 2:
                    sub = triplet[0].strip(" ")
                    obj = triplet[1].strip(" ").strip(")")
                    output[rel].append([sub, obj])
                else:
                    print("ATTENTION ! ! ! !")
                    print("Something was wrong about {}".format(triplets[i]))
            all_output[index].append(triplets)

        n1,n2,n3, n4, n5, n6 = Evaluation(gold, output)
        n_pred += n1 
        r_pred += n4
        n_gold += n2
        r_gold += n5
        n_correct += n3
        r_correct += n6
        log.info("n_pred {}. n_gold {}. n_correct {}".format(n_pred, n_gold, n_correct))
        log.info(compute_f1(n_pred, n_gold, n_correct))
        log.info("*"*30)
        log.info("r_pred {}. r_gold {}. r_correct {}".format(r_pred, r_gold, r_correct))
        log.info(compute_f1(r_pred, r_gold, r_correct))
        log.info("*"*30)
    torch.save(all_output, "all_output.pt")
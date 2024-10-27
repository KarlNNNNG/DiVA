import json
import statistics
import os
import pandas as pd
import argparse
import faiss
import sys
import math
import time
import logging

from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModel
from gpt_api import Demo, Llama
import random
import numpy as np
from gpt_evaluation import compute_f1, get_results_select

from shared.dataset import docred_reltoid, docred_idtoprompt, instance, generate_select_auto_prompt

from sklearn.metrics import classification_report
from retrieval import find_knn_example, find_lmknn_example, get_tmp_dict_roberta, get_tmp_dict_contriever
from simcse import SimCSE
import re 

import torch

def generate_relation_dict_label(dataset):
    labels = []
    with open(dataset, "r") as f:
        relation_dict = {}
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "None"
            else:
                rel = tmp_dict["relations"][0][0]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return relation_dict, labels

def generate_label(dataset, relation_dict):
    labels = []
    with open(dataset, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            if tmp_dict["relations"]== [[]]:
                rel = "NONE"
            else:
                rel = tmp_dict["relations"][0][0]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return labels



#此时文件已经处理过了，只有句子和ner, relation这些属性
#意味着doc-red也需要处理到这个程度
def get_example(example_path):
    example_dict = []
    with open(example_path, "r",encoding="utf-8") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            example_dict.append(tmp_dict)
    return example_dict
     

     
def auto_generate_example(example_dict, reltoid, idtoprompt, reasoning, demo):

    example_prompt = str()

    string = ' '.join(example_dict["document"])
    triples = "\n"
    if not reasoning:
        prompt_query = string 

    else:
        pass
    example_prompt += string
    return example_prompt

def find_prob(target, result, probs):
    try:
        index = [x.strip() for x in probs["tokens"]].index(str(target))
        #print(probs["token_logprobs"][index])
        return math.exp(probs["token_logprobs"][index])
    except:
        len_target = len(target)
        for i in range(2, len_target+1):
            for j in range(len(probs["tokens"])):
                if i + j > len(probs["tokens"]):
                    continue
                #print(j+i)
                #print(len(probs["tokens"]))
                tmp_word = "".join([probs["tokens"][x] for x in range(j, j+i)])
                if tmp_word.strip() != target:
                    continue
                else:
                    start = j
                    end = j + i
                    sum_prob = 0
                    for k in range(start, end):
                        sum_prob += math.exp(probs["token_logprobs"][k])

                    return sum_prob / i
        return 0.0

def smooth(x):
    if True:
        return np.exp(x)/sum(np.exp(x)) 
    else:
        return x
    
def compute_variance(knn_distribution):
    count_dis = [0 for x in range(len(knn_distribution))]
    for i in knn_distribution:
        count_dis[i] += 1
    tmp_distribution = 1.0 * np.array(count_dis)
    
    var = statistics.variance(tmp_distribution)
    print(var)
    if np.argmax(tmp_distribution) == 0 and var < 5:
        return 1
    else:
        return 0

def generate_lm_example(gpu_index_flat, tmp_dict, train_dict, k, reltoid, idtoprompt, args):

    knn_list = find_lmknn_example(gpu_index_flat, tmp_dict, train_dict, k)
    example_list = [train_dict[i] for i in knn_list]

    example_prompt = "Example: \n"
    
    for instance_dict in example_list:
        string = instance(instance_dict).document
        triples = "\n"
        temp = []
        if instance_dict['labels'] != []:
            for triple in instance_dict['labels']:
                if triple in temp:
                    continue
                else:
                    temp.append(triple)
                triples +=  triple[0] + "; "+ idtoprompt[int(triple[2])] +"; " +triple[1] + "\n"
        else:
           pass
        example_prompt += (string+triples)
    return example_prompt


def genetrate_random_example(train_list, k, reltoid, idtoprompt,  reasoning, demo, args):
    index = random.sample(range(len(train_list)), k)
    sample_list = [train_list[i] for i in index]
    example_prompt = str()
    temp = []
    for tmp_dict in sample_list:
        string = ' '.join(tmp_dict["document"])
        triples = "\n"
        if tmp_dict['labels'] != []:
            for triple in tmp_dict['labels']:
                if triple in temp:
                    continue
                else:
                    temp.append(triple)
                triples +=  triple[0] + ", "+ idtoprompt[int(triple[2])] +", " +triple[1] + "\n"
        if not reasoning:
            prompt_query = string 

        else:
           pass
        example_prompt += (string+triples)
    return example_prompt

def generate_knn_example(knn_model, tmp_dict, train_dict, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args):

    example_list = find_knn_example(knn_model, tmp_dict,train_dict,k)
    
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] 

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1]

        entity1 = " ".join(tmp_dict["sentences"][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the document \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list


def run(reltoid, idtoprompt, args):
    if args.model != "llama":
        demo = Demo(
                engine=args.model,
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                logprobs=1,
                )
    else:
        demo = Llama()

    
    # logger
    filetime = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    store_path = ".\\results\\knn_{}_{}".format(args.task, filetime)
    
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    log1 = get_logger('result', store_path+"/result.log")
    log2 = get_logger('negative', store_path+"/negative.log")
    
    train_list = get_example(args.example_dataset)
    dev_list = get_example(args.dev_dataset)
    test_list = get_example(args.test_dataset)

    #flat_examples = [item for sublist in test_dict.values() for item in sublist]
    #test_examples = flat_examples

   
    #knn 这一部分还没改
    if args.use_knn:
                   
        if args.retrieval == "SimCSE":
            train_sentences = [instance(x).document for x in train_list]

            knn_model = SimCSE("./sup-simcse-roberta-large")   
            knn_model.build_index(train_sentences, device="cpu")
            knn_model = torch.load("./docred_simcse.pt")

        elif args.retrieval == "Roberta":
            Roberta_path = "D:\\Projects\\gptre\\Roberta-large"
            train_sentences = [instance(x).document for x in train_list]

            index_flat = faiss.IndexFlatL2(1024)
            model = AutoModel.from_pretrained(Roberta_path)
            tokenizer = AutoTokenizer.from_pretrained(Roberta_path)

            #extractor = pipeline(model=Roberta_path, maxletask="feature-extraction")
            embed_array = []
            model.eval()
            index = 0
            for item in tqdm(train_sentences):  
                inputs = tokenizer(item, return_tensors='pt')
                if inputs["input_ids"].shape[1] > 512:
                    inputs["input_ids"] = torch.cat((inputs["input_ids"][:,0].unsqueeze(0), inputs["input_ids"][:,1:511], inputs["input_ids"][:,-1].unsqueeze(0)), dim=1).clone()
                    inputs["attention_mask"] = inputs["attention_mask"][:,0:512].clone()
                # 可能会导致dropout直接失效 torch.no_grad
                  
                result = model(**inputs)

                embeds = np.array(result[0].detach()).copy()
                embed_array.append(embeds[0,-3,:])
                if len(embed_array) % 1000 == 0:
                    print("现在保存的长度为："+str(len(embed_array)))
                    torch.save(embed_array, "docred_roberta.pt")
            print("现在保存的长度为："+str(len(embed_array)))
            torch.save(embed_array, "docred_roberta.pt")
            
            docred_roberta = torch.load("docred_roberta.pt")
            embed_list = np.array(docred_roberta)

            index_flat.add(embed_list)

        elif args.retrieval == "Contriever":
            # Contriever_path = "D:\\Projects\\gptre\\contriever"
            # train_sentences = [instance(x).document for x in train_list]
            
            # tokenizer = AutoTokenizer.from_pretrained(Contriever_path)
            # model = AutoModel.from_pretrained(Contriever_path)

            index_flat = faiss.IndexFlatL2(768)

            # embed_array = []
            # for item in tqdm(train_sentences):
            #     inputs = tokenizer(item, return_tensors='pt')
            #     if inputs["input_ids"].shape[1] > 512:
            #         inputs["input_ids"] = torch.cat((inputs["input_ids"][:,0].unsqueeze(0), inputs["input_ids"][:,1:511], inputs["input_ids"][:,-1].unsqueeze(0)), dim=1).clone()
            #         inputs["attention_mask"] = inputs["attention_mask"][:,0:512].clone()
            #         inputs["token_type_ids"] = inputs["token_type_ids"][:,0:512].clone()
            #     # 可能会导致dropout直接失效 torch.no_grad
                  
            #     result = model(**inputs)

            #     embeds = np.array(torch.mean(result[0], dim=1).squeeze(0).detach()).copy()
            #     embed_array.append(embeds)
            #     if len(embed_array) % 1000 == 0:
            #         print("现在保存的长度为："+str(len(embed_array)))
            #         torch.save(embed_array, "docred_contriever.pt")
            # torch.save(embed_array, "docred_contriever.pt")
            embed_array = torch.load("docred_contriever.pt")
            embed_list = np.array(embed_array)
            index_flat.add(embed_list)

        
    #log1.info(len(test_examples))
    log1.info(str(reltoid) )
    micro_f1 = 0.0
    
    for run in range(args.num_run):
        n_gold = n_pred = n_correct = 0
        labels = []
        preds = []
        num = 0

        if args.retrieval == "Roberta":
            tmp_dict_dic = get_tmp_dict_roberta(dev_list)
            torch.save(tmp_dict_dic, "tmp_dict_roberta.pt")
            tmp_dict_dic = torch.load("tmp_dict_roberta.pt")
        elif args.retrieval == "Contriever":
            # tmp_dict_dic = get_tmp_dict_contriever(dev_list)
            # torch.save(tmp_dict_dic, "tmp_dict_contriever.pt")
            tmp_dict_dic = torch.load("tmp_dict_contriever.pt")
        else:
            pass
        
        for tmp_dict in dev_list:
            num += 1
            tmp_knn = []
            label_other = 0

            if not args.use_knn:
                example_prompt = auto_generate_example(tmp_dict, reltoid, idtoprompt, args.reasoning, demo)
            else:
                if args.retrieval == "Roberta":
                    example_prompt = generate_lm_example(train_list, tmp_dict_dic[instance(tmp_dict).sentence], train_sentences, args.k, reltoid, idtoprompt, args.reasoning, demo, args.var, args)
                elif args.retrieval == "SimCSE":
                    example_prompt, tmp_knn, label_other, knn_list = generate_knn_example(knn_model, tmp_dict, train_list, args.k, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo, args.var, args)          
                elif args.retrieval == "Contriever": 
                    example_prompt = generate_lm_example(index_flat, tmp_dict_dic[instance(tmp_dict).document], train_list, args.k, reltoid, idtoprompt, args)
                    example_prompt += "\n Context:" + " ".join(tmp_dict['document'])
                elif args.retrieval == "Random":
                    example_prompt = genetrate_random_example(train_list, args.k, reltoid, idtoprompt, args.reasoning, demo, args)
                    example_prompt += "\n" + " ".join(tmp_dict['document'])
              
           
                                
            prompt_list = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.reasoning, args)
 
            raw_answer = get_results_select(demo, prompt_list, reltoid, idtoprompt, args)
            raw_answer = raw_answer.split("\n")
            if raw_answer[-1] == "":
                raw_answer = raw_answer[0:-1]
            raw_answer = [re.sub("[0-9]+. |- ", "", ins ) for ins in raw_answer]
            if args.evaluation:
                gold = instance(tmp_dict).labels
                entity_index = instance(tmp_dict).entity_index
                n_pred += len(raw_answer)
                n_gold += len(gold)
                log1.info(prompt_list)
                log1.info("Gold: \n ")
                log1.info(gold)
                log1.info("Answer: \n")
                #log1.info(f1_result)
                match_index = []
                raw_answer = [ins for ins in raw_answer if ins != '']
                for i in range(len(raw_answer)):
                    for j in range(len(gold)):
                        if j in match_index:
                            break
                        pred = raw_answer[i].split("; ")
                        if len(pred) != 3:
                            break
                        label = gold[j]
                        
                        p_head = re.sub(r"\"","", pred[0])
                        p_tail = re.sub(r"\"","", pred[2])
                        p_rel = re.sub(r"\"","", pred[1])

                        g_head = [v for k, v in entity_index.items() if label[0] in v]
                        g_tail = [v for k, v in entity_index.items() if label[1] in v]
                        g_rel = label[2]

                        if p_head in g_head[0] and p_tail in g_tail[0] and p_rel == g_rel and j not in match_index:
                            n_correct += 1
                            match_index.append(j)
                            log1.info("True: "+ raw_answer[i])
                            break
                    if j not in match_index:
                        log1.info("False: " + raw_answer[i])  
                log1.info(tmp_dict["labels"])
                f1_result = compute_f1(n_pred, n_gold, n_correct)
                log1.info(str(f1_result))
                log1.info("processing:" + str(100*num / len(test_list) ) + "%")
                log1.info("\n--------------------------------------------------------------------------------------------")
                
            else:
                log1.info(prompt_list)
                log1.info("Raw Answer:\n" + raw_answer)
                log1.info("Gold:\n" + instance(tmp_dict).labels)
                log1.info(tmp_dict["title"])
                log1.info("processing:" + str(100*num / len(test_list) ) + "%")
                log1.info("\n--------------------------------------------------------------------------------------------")


        if args.evaluation:
            #report = classification_report(labels, preds, digits=4,output_dict=True)
            #log1.info(report)
            #micro_f1 += f1_result["f1"]
            
            # if args.store_error_reason:
            #     with open("stored_reason/{}_dev.txt".format(args.task), "w") as f:
            #         json.dump(store_error_reason, f)
            # with open("{}/labels.csv".format(store_path), "w") as f:
            #     f.write('\n'.join([str(labels)]))
            # with open("{}/preds.csv".format(store_path), "w") as f:
            #     f.write('\n'.join([str(preds)]))
            # with open("{}/probs.csv".format(store_path), "w") as f:
            #     for prob in whole_prob:
            #         json.dump(prob, f)
            #         f.write("\n")
            # with open("{}/prob_on_rel.csv".format(store_path), "w") as f:
            #     f.write('\n'.join([str(x) for x in whole_prob_on_rel]))
            # micro_f1 += f1_result["f1"]
            # with open("{}/azure_error.csv".format(store_path), "w") as f:
            #     f.write('\n'.join([str(azure_error)]))
            # with open("{}/knn.csv".format(store_path), "w") as f:
            #     for line in whole_knn:
            #         f.write('\n'.join([str(line)]))
            #         f.write("\n")
            # df = pd.DataFrame(report).transpose()
            # df.to_csv("{}/result_per_rel.csv".format(store_path))
            # #print(report)
            # print(azure_error)
            # #assert False
        #avg_f1 = micro_f1 / args.num_run
        #log1.info("AVG f1:", avg_f1)
            log1.info(args)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True, choices=["ace05","semeval","tacred","scierc","wiki80","docred"])
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--example_dataset", type=str, default=None, required=True)
    parser.add_argument("--dev_dataset", type=str, default=None, required=True)
    parser.add_argument("--test_dataset", type=str, default=None, required=True)
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_label", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--use_knn", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--structure", type=int, default=0)
    parser.add_argument("--retrieval", type=str, default="SimCSE")
    parser.add_argument("--evaluation", type=str, default=False)

    args = parser.parse_args()

    if args.reasoning == 1:
        args.reasoning = True
    else:
        args.reasoning = False

    print(args)

    random.seed(args.seed)

    run(docred_reltoid,docred_idtoprompt, args)

  



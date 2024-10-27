from simcse import SimCSE
from shared.dataset import instance
from transformers import pipeline
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
#model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


#embeddings = model.encode("A woman is reading.")

#sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
#sentences_b = ['He plays guitar.', 'A woman is making a photo.']
#similarities = model.similarity(sentences_a, sentences_b)
#print(similarities)


#sentences = ['A woman is reading.', 'A man is playing a guitar.']
#model.build_index(sentences)
#results = model.search("He plays guitar.")

#print(results)


def find_knn_example(model, test_dict, train_dict, k, entity_info):
    if entity_info:
        test_sentences = instance(test_dict).reference
    else:
        test_sentences = " ".join(test_dict["sentences"])
    test_id = test_dict["doc_key"]
    label_other = 0
    #train_dict = {" ".join(x["sentences"]):x for x in train_list}
    #train_sentences = [x for x in train_dict.keys()]
    
    #print(len(test_sentences))
    #print(len(train_sentences))
    #model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    #model.build_index(train_sentences, device="cpu")

    #for x in test_sentences:

    #    knn_result = model.search(x, device="cpu", threshold=0.3, top_k=3)
    #    print(knn_result)
    #    assert False
    knn_result = model.search(test_sentences, device="cpu", threshold=0.0, top_k=k)
    #print(knn_result)
    knn_list = [train_dict[x[0]] for x in knn_result]
    #if var and not no_na:
    #    label_other = knn_variance(knn_list)

    #print(train_sentences[0])
    #print(knn_list)
    #assert False
    return knn_list


#为了获得开发集或者是测试集的特征
def get_tmp_dict_roberta(test_dict):
    Roberta_path = "D:\\Projects\\gptre\\Roberta-large"
    dev_dict  = {}
    model = AutoModel.from_pretrained(Roberta_path)
    tokenizer = AutoTokenizer.from_pretrained(Roberta_path)
    
    for example in tqdm(test_dict):
        test_sentence = instance(example).documents
        inputs = tokenizer(test_sentence, return_tensors='pt')
        if inputs["input_ids"].shape[1] > 512:
            inputs["input_ids"] = torch.cat((inputs["input_ids"][:,0].unsqueeze(0), inputs["input_ids"][:,1:511], inputs["input_ids"][:,-1].unsqueeze(0)), dim=1).clone()
            inputs["attention_mask"] = inputs["attention_mask"][:,0:512].clone()
        result = model(**inputs)
        embeds = result[0].detach().numpy().copy()
        xq = embeds[:,-3,:]
        dev_dict[instance(example).sentence] = xq
    return dev_dict

def get_tmp_dict_contriever(test_dict):
    contriever_path = "D:\\Projects\\gptre\\contriever"
    dev_dict  = {}
    model = AutoModel.from_pretrained(contriever_path)
    tokenizer = AutoTokenizer.from_pretrained(contriever_path)
    
    for example in tqdm(test_dict):
        test_sentence = instance(example).document
        inputs = tokenizer(test_sentence, return_tensors='pt')
        if inputs["input_ids"].shape[1] > 512:
            inputs["input_ids"] = torch.cat((inputs["input_ids"][:,0].unsqueeze(0), inputs["input_ids"][:,1:511], inputs["input_ids"][:,-1].unsqueeze(0)), dim=1).clone()
            inputs["attention_mask"] = inputs["attention_mask"][:,0:512].clone()
            inputs["token_type_ids"] = inputs["token_type_ids"][:,0:512].clone()

        result = model(**inputs)
        embeds = np.array(torch.mean(result[0], dim=1).detach()).copy()
        assert embeds.shape[1] == 768
        dev_dict[test_sentence] = embeds
    return dev_dict

def find_lmknn_example(gpu_index_flat, xq, train_dict, k):

    D, I = gpu_index_flat.search(xq, k)

    knn_list = I[0,:k]

    return knn_list


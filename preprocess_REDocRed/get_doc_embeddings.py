from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
import argparse
import os

def main(path_doc, write_file):

	model = SentenceTransformer('all-mpnet-base-v2')
	model.max_seq_length=512

	print("model is loaded")

	with open(path_doc, "rb") as f:
		data = json.load(f)

	list_pars = [doc["paragraph"] for doc in data]
	list_par_ids = [i for i in range(len(data))]
	list_par_titles = [doc["title"] for doc in data]

	embeddings = model.encode(list_pars, show_progress_bar=True, normalize_embeddings=True, batch_size=128)

	#We also keep the length of each doc (i.e. num_words) in case we need a filtering later on 
	num_words = np.array(list(map(lambda x: sum([len(l) for l in x['sents']]), data)))

	with open(write_file, 'wb') as fOut:
		pickle.dump({'paragraph_ids':list_par_ids, 'paragraph_titles':list_par_titles, 'embeddings':embeddings, 'num_words':num_words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-p', '--path_doc', type=str, default="train_annotated_preprocessed.json", help='full path to the doc files kept in paragraph format')
    # parser.add_argument('-w', '--write_file', type=str, default="embeddings_train_annotated.pkl", help='Name of the postprocess log file')
    

    # args = parser.parse_args()

	#main("D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\train_revised_preprocessed.json" ,  "dataset\\redocred\\data\\embeddings_train.pkl")
	#main("D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\test_revised_preprocessed.json" ,  "dataset\\redocred\\data\\embeddings_test.pkl")
	main("D:\\Projects\\GenerativeRE\\dataset\\redocred\\data\\dev_revised_preprocessed.json" ,  "dataset\\redocred\\data\\embeddings_dev.pkl")
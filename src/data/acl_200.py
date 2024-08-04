import random
import re
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from src import config as cfg


class ACL200Dataset(Dataset):
    def __init__(self, corpus = [], paper_database = {}, context_database = {},
                 tokenizer = None, 
                 rerank_top_K = 2000,
                 max_input_length = 512, 
                 padding = "max_length", 
                 truncation=True,
                 sep_token = "<sep>",
                 cit_token = "<cit>",
                 eos_token = "<eos>",
                 n_document = 32,
                 max_n_positive = 1,
                 mode='train'
              ):
        ## structure of the corpus
        if mode == 'train':
            n_samples = cfg['n_training_examples_wt_prefetched_ids_for_reranking']
            self.corpus = corpus[:n_samples]
        elif mode == 'val':
            n_samples = cfg['n_valid_examples_wt_prefetched_ids_for_reranking']
            self.corpus = corpus[:n_samples]
        else:
            self.corpus = corpus

        self.paper_database = paper_database
        self.context_database = context_database
        self.tokenizer = tokenizer
        self.rerank_top_K = rerank_top_K
        self.max_input_length = max_input_length
        self.padding = padding
        self.truncation = truncation
        self.sep_token = sep_token
        self.cit_token = cit_token
        self.eos_token = eos_token
        self.special_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.mode = mode
        self.n_document = n_document
        self.max_n_positive = max_n_positive

        self.irrelevance_level_for_positive = 0
        self.irrelevance_level_for_negative = 1
        
    def __len__(self):
        return len(self.corpus)
    
    def get_paper_text(self, paper_id):
        paper_info = self.paper_database.get(paper_id, {})
        title = paper_info.get("title","")
        abstract = paper_info.get("abstract", "")
        return title + " " + abstract
    
    @staticmethod
    def year_from_id(paper_id):
        digits = int(paper_id[1:3])
        return 2000 + digits if digits < 60 else 1900 + digits

    def get_random_prefetched_samples(self, context_id):
        citing, cited = context_id.split('_')[:2]
        random_papers = set()
        while len(random_papers) < cfg['num_negs']:
            paper_ids = random.sample(self.paper_database.keys(), k=cfg['num_negs'])
            for pid in paper_ids:
                if pid in [citing, cited]:
                    continue
                if self.year_from_id(pid) > self.year_from_id(citing):  # skip new papers
                        continue
                random_papers.add(pid)
        return list(random_papers)[:cfg['num_negs']]
    
    def __getitem__(self, idx):
        ## step 1: get the query information, based on local or global citation recommendation 
        ## step 2: get the candidate documents
        ## step 3: construct the input to the scorer model
        data = self.corpus[idx]

        context_id = data["context_id"]
        citing_id = self.context_database[context_id]["citing_id"]
        context_text = self.context_database[context_id]["masked_text"].replace("TARGETCIT", self.cit_token)
        citing_text = self.get_paper_text(citing_id)
        
        if self.mode == 'train':
            prefetched_ids = self.get_random_prefetched_samples(context_id)
        elif self.mode == 'test':
            prefetched_ids = data['prefetched_ids']
        elif self.mode == 'val':
            prefetched_ids = data['positive_ids'] + data['prefetched_ids']

        prefetched_ids = prefetched_ids[:self.rerank_top_K]
            
        positive_ids = data['positive_ids']
        positive_ids_set = set(positive_ids)
        
        negative_ids = list(set(prefetched_ids) - set(positive_ids + [citing_id]))

        if self.mode == 'train':
            ## sample up to max_n_positive positive ids
            positive_id_indices = np.arange(len(positive_ids))
            np.random.shuffle(positive_id_indices)
            candidate_id_list = [positive_ids[i]  for i in positive_id_indices[:self.max_n_positive]]
            irrelevance_levels_list = [self.irrelevance_level_for_positive] * len(candidate_id_list)  

            for pos in np.random.choice(len(negative_ids), self.n_document - len(candidate_id_list)):
                irrelevance_levels_list.append(self.irrelevance_level_for_negative)
                candidate_id_list.append(negative_ids[pos])
            irrelevance_levels_list = np.array(irrelevance_levels_list).astype(np.float32)

        elif self.mode in ['val', 'test']:
            candidate_id_list = prefetched_ids
            irrelevance_levels_list = np.array([self.irrelevance_level_for_positive if candidate_id in positive_ids_set  
                                                else self.irrelevance_level_for_negative 
                                                for candidate_id in candidate_id_list]).astype(np.float32)
        query_text_list = []
        candidate_text_list = []
        
        for candidate_id in candidate_id_list:
            candidate_text = self.get_paper_text(candidate_id)
            
            query_text_list.append(" ".join(citing_text.split()[:int(self.max_input_length * 0.35)]) + self.sep_token + context_text)
            candidate_text_list.append(candidate_text)


        encoded_seqs = self.tokenizer(query_text_list,candidate_text_list,  
                                      max_length = self.max_input_length, 
                                      padding = self.padding, 
                                      truncation = self.truncation)
        
        for key in encoded_seqs:
            encoded_seqs[key] = np.asarray(encoded_seqs[key])
        
        encoded_seqs.update({
                "irrelevance_levels": irrelevance_levels_list,
                "num_positive_ids": len(positive_ids),
                "candidate_ids": candidate_id_list,
                'context_id': context_id
              })

        return encoded_seqs
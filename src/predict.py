import os
import json
import time
import argparse
import numpy as np
import torch.nn as nn
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src import config as cfg
from src.utils.data_utils import *
from src.models.model import Scorer
from transformers import AutoTokenizer
from src.data.preprocess import reformat
from src.data.acl_200 import ACL200Dataset

from src.utils.logger import logger, set_logger_tag

set_logger_tag(tag='PREDICTION')


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="Path to the best model")
    parser.add_argument('--out_dir', help="Path to the output directory")


if __name__ == "__main__":
    args = cli()

    tokenizer = AutoTokenizer.from_pretrained(cfg['initial_model_path'])
    tokenizer.add_special_tokens({ 'additional_special_tokens': ['<cit>','<sep>','<eos>'] })
    
    test_corpus = reformat(cfg['test_corpus_path'], mode='train')

    paper_database = json.load(open(cfg['paper_database_path']))
    
    context_database = json.load(open(cfg['context_database_path']))

        
    rerank_dataset = ACL200Dataset(test_corpus, paper_database, context_database, tokenizer,
                                  rerank_top_K = cfg['rerank_top_K'],
                                  max_input_length = cfg['max_input_length'],
                                  mode = 'test',
                                  n_document= cfg['n_document'], 
                                  max_n_positive = cfg['max_n_positive'],
                                 )
    rerank_dataloader = DataLoader(rerank_dataset, batch_size= cfg['n_query_per_batch'], shuffle= True, 
                                  num_workers= cfg['num_workers'],  drop_last= True, 
                                  worker_init_fn = lambda x:[np.random.seed(int(time.time()) + x), torch.manual_seed(int(time.time()) + x)],
                                  pin_memory= True)

    vocab_size = len(tokenizer)
    scorer = Scorer(cfg['initial_model_path'], vocab_size)

    if os.path.exists(args.model_path):
        ckpt = load_model(args.model_path)
        scorer.load_state_dict(ckpt["scorer"])
        logger.info("model loaded!")
    
    if cfg['gpu_list'] is not None:
        assert len(cfg['gpu_list']) == cfg['n_device']
    else:
        cfg['gpu_list'] = np.arange(cfg['n_device']).tolist()

    device = torch.device("cuda:%d"%(cfg['gpu_list'][0]) if torch.cuda.is_available() else "cpu")
    scorer.to(device)
    
    if cfg['gpu_list'] is not None:
        assert len(args['gpu_list']) == cfg['n_device']
    else:
        args['gpu_list'] = np.arange(cfg['n_device']).tolist()
    device = torch.device( "cuda:%d"%(cfg['gpu_list'][0]) if torch.cuda.is_available() else "cpu")
    scorer.to(device)

    if device.type == "cuda" and args.n_device > 1:
        scorer = nn.DataParallel( scorer, args['gpu_list'])

    context_scores = defaultdict(list)

    for count, batch in enumerate(tqdm(rerank_dataloader)):
        irrelevance_levels = batch["irrelevance_levels"].to(device)
        input_ids =  batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_positive_ids = batch["num_positive_ids"] 
        positive_ids = batch["positive_ids"]
        context_ids = batch["context_id"]
        n_doc = input_ids.size(1)

        input_ids = input_ids.view(-1,input_ids.size(2))
        token_type_ids = token_type_ids.view(-1,token_type_ids.size(2))
        attention_mask = attention_mask.view(-1, attention_mask.size(2))

        score = []
        for pos in range(0, input_ids.size(0), args.sub_batch_size):
            with torch.no_grad():
                score.append(scorer( 
                    {
                        "input_ids":input_ids[pos:pos+args.sub_batch_size],
                        "token_type_ids":token_type_ids[pos:pos+args.sub_batch_size],
                        "attention_mask":attention_mask[pos:pos+args.sub_batch_size]
                    }).detach())
        score = torch.cat(score, dim =0).view(-1, n_doc).cpu().numpy()
        for j, (cid, pid) in enumerate(zip(context_ids, positive_ids)):
            context_scores[cid].append((pid, score[j]))
    
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(context_scores, os.path.join(args.out_dir, 'predictions.json'), ensure_ascii=False)
    logger.info("Done!")

        

    

    
import os
import json
import time
import glob
import argparse
import numpy as np
from tqdm import tqdm 
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src import config as cfg
from src.utils.data_utils import *
from src.models.model import Scorer
from transformers import AutoTokenizer
from src.data.preprocess import reformat
from src.data.acl_200 import ACL200Dataset

from src.utils.logger import logger, set_logger_tag

set_logger_tag(logger, tag='PREDICTION')


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help="Path to the best model")
    parser.add_argument("--test_dir", help="Path to testing folder")
    parser.add_argument('--out_dir', help="Path to the output directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = cli()

    tokenizer = AutoTokenizer.from_pretrained(cfg['initial_model_path'])
    tokenizer.add_special_tokens({ 'additional_special_tokens': ['<cit>','<sep>','<eos>'] })
    vocab_size = len(tokenizer)
    scorer = Scorer(cfg['initial_model_path'], vocab_size)

    if os.path.exists(args.model_dir):
        ckpt = load_model(args.model_dir)
        scorer.load_state_dict(ckpt["scorer"])
        logger.info("model loaded!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer.to(device)
    
    for i, test_path in enumerate(glob(os.path.join(args.test_dir, "*.json"))):
        test_corpus = reformat(test_path, mode='test')
        paper_database = json.load(open(cfg['paper_database_path']))
        context_database = json.load(open(cfg['context_database_path']))

        dataset = ACL200Dataset(test_corpus, paper_database, context_database, tokenizer,
                                    rerank_top_K=cfg['rerank_top_K'],
                                    max_input_length = cfg['max_input_length'],
                                    mode = 'test',
                                    max_n_positive = cfg['max_n_positive'],
                                    )
        dataloader = DataLoader(dataset, batch_size= cfg['n_query_per_batch'], shuffle= False, 
                                    num_workers= cfg['num_workers'],  drop_last= False, 
                                    worker_init_fn = lambda x:[np.random.seed(int(time.time()) + x), torch.manual_seed(int(time.time()) + x)],
                                    pin_memory= True)


        context_scores = defaultdict(list)

        for count, batch in enumerate(tqdm(dataloader)):
            irrelevance_levels = batch["irrelevance_levels"].to(device)
            input_ids =  batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_positive_ids = batch["num_positive_ids"] 
            candidate_ids = batch["candidate_ids"]
            context_id = batch['context_id']
            n_doc = input_ids.size(1)

            input_ids = input_ids.view(-1,input_ids.size(2))
            token_type_ids = token_type_ids.view(-1,token_type_ids.size(2))
            attention_mask = attention_mask.view(-1, attention_mask.size(2))

            score = []
            for pos in range(0, input_ids.size(0), cfg['eval_sub_batch_size']):
                with torch.no_grad():
                    score.append(scorer( 
                        {
                            "input_ids":input_ids[pos:pos+cfg['eval_sub_batch_size']],
                            "token_type_ids":token_type_ids[pos:pos+cfg['eval_sub_batch_size']],
                            "attention_mask":attention_mask[pos:pos+cfg['eval_sub_batch_size']]
                        }).detach())
            score = torch.cat(score, dim=0).cpu().tolist()
            for j, can_id in enumerate(candidate_ids):
                context_scores[context_id[0]].append((can_id[0], score[j]))
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(context_scores, open(os.path.join(args.out_dir, f'predictions_{i+1}.json'), 'wt'), ensure_ascii=False)
        logger.info("Done!")

        

    

    
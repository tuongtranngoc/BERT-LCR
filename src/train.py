import os
import json
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader 

from src import config as cfg
from src.utils.data_utils import *
from src.models.model import Scorer
from transformers import AutoTokenizer
from src.data.preprocess import reformat
from src.utils.losses import TripletLoss
from src.data.acl_200 import ACL200Dataset


from src.utils.logger import logger, set_logger_tag

set_logger_tag(logger, tag="TRAINING")


def train_iteration(batch):
    irrelevance_levels = batch["irrelevance_levels"].to(device)
    input_ids =  batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    n_doc = input_ids.size(1)
    score = scorer(
        {
            "input_ids":input_ids.view(-1, input_ids.size(2)),
            "token_type_ids":token_type_ids.view(-1, token_type_ids.size(2)),
            "attention_mask":attention_mask.view(-1, attention_mask.size(2))
        })
    score = score.view(-1, n_doc)
    loss = triplet_loss(score, irrelevance_levels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    return loss.item()

def validate_iteration(batch):
    irrelevance_levels = batch["irrelevance_levels"].to(device)
    input_ids =  batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    n_doc = input_ids.size(1)
    with torch.no_grad():
        score = scorer({
            "input_ids":input_ids.view(-1, input_ids.size(2)),
            "token_type_ids":token_type_ids.view(-1, token_type_ids.size(2)),
            "attention_mask":attention_mask.view(-1, attention_mask.size(2))
        })
        score = score.view(-1, n_doc)
        loss = triplet_loss(score, irrelevance_levels)
    return loss.item()


if __name__ == "__main__":

    if not os.path.exists(cfg['model_folder']):
        os.makedirs(cfg['model_folder'])
    if not os.path.exists(cfg['log_folder']):
        os.makedirs(cfg['log_folder'])

    # restore most recent checkpoint
    if cfg['restore_old_checkpoint']:
        ckpt = load_model(cfg['model_folder'])
    else:
        ckpt = None

    tokenizer = AutoTokenizer.from_pretrained(cfg['initial_model_path'])
    tokenizer.add_special_tokens({ 'additional_special_tokens': ['<cit>','<sep>','<eos>'] })
    
    train_corpus = reformat(cfg['train_corpus_path'], mode='train')

    paper_database = json.load(open(cfg['paper_database_path']))
    
    context_database = json.load(open(cfg['context_database_path']))

        
    rerank_dataset = ACL200Dataset(train_corpus, paper_database, context_database, tokenizer,
                                  rerank_top_K = cfg['rerank_top_K'],
                                  max_input_length = cfg['max_input_length'],
                                  mode = 'train',
                                  n_document= cfg['n_document'], 
                                  max_n_positive = cfg['max_n_positive'],
                                 )
    rerank_dataloader = DataLoader(rerank_dataset, batch_size= cfg['n_query_per_batch'], shuffle= True, 
                                  num_workers= cfg['num_workers'],  drop_last= True, 
                                  worker_init_fn = lambda x:[np.random.seed(int(time.time()) + x), torch.manual_seed(int(time.time()) + x)],
                                  pin_memory= True)

    val_corpus = reformat(cfg['val_corpus_path'], mode='val')

    val_rerank_dataset = ACL200Dataset(val_corpus, paper_database, context_database, tokenizer,
                                  rerank_top_K = cfg['rerank_top_K'],
                                  max_input_length = cfg['max_input_length'],
                                  mode = 'val',
                                  n_document= cfg['n_document'], 
                                  max_n_positive = cfg['max_n_positive'],
                                 )
    val_rerank_dataloader = DataLoader(val_rerank_dataset, batch_size= cfg['n_query_per_batch'], shuffle= False, 
                                  num_workers= cfg['num_workers'],  drop_last= True, 
                                  worker_init_fn = lambda x:[np.random.seed(int(time.time())+x), torch.manual_seed(int(time.time()) + x)],
                                  pin_memory= True)

    vocab_size = len(tokenizer)
    scorer = Scorer(cfg['initial_model_path'], vocab_size)

    if ckpt is not None:
        scorer.load_state_dict(ckpt["scorer"])
        logger.info("model restored!")
    
    if cfg['gpu_list'] is not None:
        assert len(cfg['gpu_list']) == cfg['n_device']
    else:
        cfg['gpu_list'] = np.arange(cfg['n_device']).tolist()

    device = torch.device("cuda:%d"%(cfg['gpu_list'][0]) if torch.cuda.is_available() else "cpu" )
    scorer.to(device)

    if device.type == "cuda" and cfg['n_device'] > 1:
        scorer = nn.DataParallel(scorer, cfg['gpu_list'])
        model_parameters = [ par for par in scorer.module.parameters() if par.requires_grad  ] 
    else:
        model_parameters = [ par for par in scorer.parameters() if par.requires_grad  ] 
    optimizer = torch.optim.AdamW(model_parameters , lr= float(cfg['initial_learning_rate']),  weight_decay = cfg['l2_weight'] ) 

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("optimizer restored!")

    current_batch = 0
    if ckpt is not None:
        current_batch = ckpt["current_batch"]
        logger.info("current_batch restored!")
    running_losses = []

    triplet_loss = TripletLoss(cfg['base_margin'])
    for epoch in range(cfg['num_epochs']):
        for count, batch in enumerate(tqdm(rerank_dataloader)):
            current_batch +=1

            loss = train_iteration(batch)

            running_losses.append(loss)

            if current_batch % cfg['print_every'] == 0:
                logger.info("[batch: %05d] loss: %.4f"%(current_batch, np.mean(running_losses)))
                os.system("nvidia-smi > %s/gpu_usage.log"%(cfg['log_folder']))
                running_losses = []
            if current_batch % cfg['save_every'] == 0 :  
                save_model( { 
                    "current_batch":current_batch,
                    "scorer": scorer,
                    "optimizer": optimizer.state_dict()
                    } ,  cfg['model_folder']+"/model_batch_%d.pt"%(current_batch), 10)
                logger.info("Model saved!")

            if current_batch % cfg['validate_every'] == 0:
                running_losses_val = []
                for val_count, batch in enumerate(tqdm(val_rerank_dataloader)):
                    loss = validate_iteration(batch)
                    running_losses_val.append(loss)

                    if val_count >= cfg['num_validation_iterations']:
                        break
                logger.info("[batch: %05d] validation loss: %.4f"%(current_batch, np.mean(running_losses_val)))                

        running_losses_val = []
        for val_count, batch in enumerate(tqdm(val_rerank_dataloader)):
            loss = validate_iteration(batch)
            running_losses_val.append(loss)
            if val_count >= cfg['num_validation_iterations']:
                break
        logger.info("[batch: %05d] validation loss: %.4f"%(current_batch, np.mean(running_losses_val)))

        save_model({ 
                    "current_batch":current_batch,
                    "scorer": scorer,
                    "optimizer": optimizer.state_dict()
                    },  
                    cfg['model_folder']+"/model_batch_%d.pt"%(current_batch), cfg['max_num_checkpoints'])
        logger.info("Model saved!")
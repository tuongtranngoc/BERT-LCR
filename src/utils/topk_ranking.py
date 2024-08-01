
import os
import json
import argparse
from tqdm import tqdm
from cache_decorator import Cache

import spacy
from rank_bm25 import BM25Okapi


nlp = spacy.load('en_core_web_sm')
cache_dir='.cache/bm25_ranking'
os.makedirs(cache_dir, exist_ok=True)


def tokenize(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc]


@Cache(cache_dir=cache_dir, use_source_code=False)
def context_tokenizer(contexts, papers):
    tokenized_contexts = []
    for cid in tqdm(contexts):
        c = contexts[cid]
        citing_title_abstract = papers[c['citing_id']]['title'] + ' ' + papers[c['citing_id']]['abstract']
        context = c['masked_text']
        tokenized_contexts.append((tokenize(citing_title_abstract + ' ' + context), cid))

    return tokenized_contexts


@Cache(cache_dir=cache_dir, use_source_code=False)
def paper_tokenizer(papers):
    return {
        k: tokenize(papers[k]['title'] + ' ' + papers[k]['abstract'])
        for k in tqdm(papers)
    }


def main(args):
    contexts = json.load(open(args.contexts_file))
    context_hardnegatives = {}

    # preprocess all papers
    papers = json.load(open(args.papers_file))
    print('Tokenizing papers...')
    _tokenized_papers = paper_tokenizer(papers)
    print(f'Tokenized {len(_tokenized_papers)} papers')
    
    tokenized_papers, pids = [], []
    for pid in _tokenized_papers:
        tokenized_papers.append(_tokenized_papers[pid])
        pids.append(pid)

    # preprocess contexts
    print('Tokenizing contexts...')
    tokenized_contexts = context_tokenizer(contexts, papers)
    print(f'Tokenized {len(tokenized_contexts)} contexts')

    # fit BM25 on papers
    model = BM25Okapi(tokenized_papers)
    
    # get BM25 scores for contexts
    for context, cid in tokenized_contexts:
        citing, cited = cid.split('_')[:2]
        scores = model.get_scores(context)
        
        # sort pids by scores and keep top k + 1
        sorted_pids = [pid for pid, _ in sorted(list(zip(pids, scores)), key=lambda x: x[1], reverse=True)][:args.k+1]
        
        # remove citing id if in candidates
        if citing in sorted_pids:
            sorted_pids.remove(citing)
        sorted_pids = sorted_pids[:args.k]
         
        # add cited id if not in candidates
        if cited not in sorted_pids:
            sorted_pids[-1] = cited
        
        context_hardnegatives[cid] = sorted_pids

    json.dump(context_hardnegatives, open(args.output_file, 'wt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts_file', help='JSON file with context texts')
    parser.add_argument('--papers_file', help='JSON file with paper title and abstracts')
    parser.add_argument('--output_file', help='JSON file in which dictionary of context ids as keys and list of paper ids as values is written')
    parser.add_argument('--k', type=int, default=2000, help='number of candidates to produce for each context')
    args = parser.parse_args()

    main(args)
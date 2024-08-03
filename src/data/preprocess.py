import pandas as pd
import json
import os


def reformat(data_path, mode='train'):
    data = json.load(open(data_path))
    data = pd.DataFrame(data)

    if mode == 'val':
        data = data.groupby(['context_id']).agg({'true_ref':list, 'neg_ref':list}).reset_index()
        data.true_ref = data.true_ref.map(lambda x: list(set(x)))
        data.neg_ref = data.neg_ref.map(lambda x: list(set(x)))
        data = data.rename(columns={'true_ref': 'positive_ids', 'neg_ref':'prefetched_ids'})
    
    elif mode == 'test':
        data = data.groupby(['context_id']).agg({'paper_id':list}).reset_index()
        data.paper_id = data.paper_id.map(lambda x: list(set(x)))
        data['positive_ids'] = data.context_id.map(lambda x: [x.split('_')[1]])
        data = data.rename(columns={'paper_id':'prefetched_ids'})
    
    elif mode == 'train':
        data.paper_id = data.paper_id.map(lambda x: [x])
        data = data.rename(columns={'paper_id': 'positive_ids'})

    return data.to_dict("records")
        

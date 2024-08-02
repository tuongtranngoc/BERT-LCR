import os
import json
import argparse
import pandas as pd

from src.utils.multiprocess import run_multiprocess_tqdm


def convert(data_path, save_dir):
    basename = os.path.splitext(os.path.basename(data_path))[0]
    if save_dir is None:
        save_dir = os.path.dirname(data_path)
    os.makedirs(save_dir, exist_ok=True)
    data = json.load(open(data_path, encoding='utf-8'))
    data = pd.DataFrame(data)
    context_ids = data.context_id.unique()
    
    global single_process
    
    def single_process(context_id):
        paper_ids = data[data.context_id==context_id].paper_id.tolist()
        return {
            'context_id': context_id,
            'positive_ids': context_id.split('_')[1],
            'prefetched_ids': paper_ids
        }
    
    constructed_data = run_multiprocess_tqdm(single_process, context_ids[:100], num_processes=8, chunk_size=10)
    json.dump(constructed_data, open(os.path.join(save_dir, basename + '_with_prefetched_bm25.json'), 'w'), ensure_ascii=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='JSON file that need to convert')
    parser.add_argument('--out_dir', type=str, default=None, help="Path to ouput folder")
    
    args = parser.parse_args()
    
    convert(args.data_path, args.out_dir)
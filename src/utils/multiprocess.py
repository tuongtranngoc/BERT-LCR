import functools
from tqdm.contrib.concurrent import process_map


def run_multiprocess_tqdm_wrapers(func, inputs):
    return func(*inputs)


def run_multiprocess_tqdm(func, *inputs, num_processes=3, chunk_size=1):
    partial_fn = functools.partial(run_multiprocess_tqdm_wrapers, func)
    return process_map(partial_fn, list(zip(*inputs)), max_workers=num_processes, chunksize=chunk_size)

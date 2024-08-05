# Local Citation Recommendation using SciBERT

<p align="center">
    <image src="images/scibert.png">
</p>

The project aims to find relevant articles to the given citation context. Inspired by [LCR with Hierarchical-Attention Text Encoder and SciBERT-based Reranking](https://arxiv.org/pdf/2112.01206), I used the Reranking stage to query relevant articles without the prefetching stage.


## Environment
The dependencies are listed in `requirements.txt`. Please install and follow the command below:

```bash
pip install -r requirements.txt
```

## Data Preparation
We provide links for downloading preprocessed dataset instances as well as training, validation, and test splits. The ACL-200 dataset files are compressed and available on the following link [ACL-200](https://drive.google.com/file/d/1i-0cmwTM7rBL937PoPBK3mFLvGBusJLS/view?usp=sharing)


## Training
Before training, you can config parameters in the file `src/configs/scibert.yaml`
Then, Please run this command:

```shell
python -m src.train
```

## Evaluation
The testing set is so large (articles from 2015) that we only suggest testing with 100 examples randomly.

First, please run the `prediction.py` file to get the score results 
```shell
python -m src.predict --model_dir <path_to_model_directory> --out_dir <path_to_result_output_file> --test_file dataset/acl_200/test_100.json
```

Next, to get the evaluation result, please run the command:
```shell
python -m src.evaluate --pred_file <path_to_result_output_file>
```

## References
+ [Improved Local Citation Recommendation Based on Context Enhanced with Global Information](https://github.com/zoranmedic/DualLCR)
+ [Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking](https://github.com/nianlonggu/Local-Citation-Recommendation)
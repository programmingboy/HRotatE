
# HRotatE: Hybrid Relational Rotation Embedding for Knowledge Graph
**Introduction**

This is the PyTorch implementation of the [HRotatE] model for knowledge graph embedding (KGE). 

This code is updated version of RotatE code(https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)


**Implemented features**

Models:
 - [x] HRotatE
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

For example, this command train a HRotatE model on FB15k dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 128 -b 256 -d 1000 \
 -g 24.0 -a 1.5 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

To reprocude the best results, you can run the bash commands in best_config.sh to get the best performance of HRotatE, RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Yago3-10).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train HRotatE FB15k 0 0 256 128 1000 24.0 1.5 0.0001 150000 16 -de


**Results of the HRotatE model**

| Dataset | FB15k | FB15k-237 | WN18 | WN18RR | Yago3-10
|-------------|-------------|-------------|-------------|-------------|-------------|
| MRR | .799 | .338  | .951  |.483 | .497 |
| HITS@1 | .751 | .243 | .945 | .438 | .399 |
| HITS@3 | .833 | .373 | .954 | .499 | .554 |
| HITS@10 | .832 | .530 | .960 | .572 | .681 |

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.


# MPComplEx

This is the data and coded for our ICAART 2025 paper **Improving Temporal Knowledge Graph Completion via Tensor Decomposition with Relation-Time Context and Multi-time Perspective**

## Installation

Create a conda environment with pytorch and scikit-learn :

```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment

```
python setup.py install
```

## Datasets

To download the datasets, go to the ./tkbc/scripts folder and run:

```
chmod +x download_data.sh
./download_data.sh
```

GDELT dataset can be download [here](https://github.com/BorealisAI/de-simple/tree/master/datasets/gdelt) and rename the files without ".txt" suffix.

Once the datasets are downloaded, add them to the package data folder by running :

```
python process_icews.py
python process_yago.py
python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

Run the following commands to reproduce the results

**TPComplEx**

```sh
python learner.py --cuda --dataset ICEWS14 --model MPComplEx --rank 1594 --emb-reg-type N3 --emb-reg 1e-2 --time-reg-type Lambda3 --time-reg 1e-2

python learner.py --cuda --dataset ICEWS05-15 --model MPComplEx --rank 886 --emb-reg-type N3 --emb-reg 0.01 --time-reg-type Lambda3 --time-reg 0.01

python learner.py --cuda --dataset yago15k --model MPComplEx --rank 1892 --no_time_emb --emb-reg-type N3 --emb-reg 0.1 --time-reg-type Lambda3 --time-reg 1

python learner.py --cuda --dataset gdelt --model MPComplEx --rank 1256 --emb-reg-type N3 --emb-reg 1e-4 --time-reg-type Lambda3 --time-reg 1e-2
```

**TPComplEx**

```sh
python learner.py --cuda --dataset ICEWS14 --model TPComplEx --rank 1594 --emb-reg-type N3 --emb-reg 1e-2 --time-reg-type Lambda3 --time-reg 1e-2

python learner.py --cuda --dataset ICEWS05-15 --model TPComplEx --rank 886 --emb-reg-type N3 --emb-reg 0.01 --time-reg-type Lambda3 --time-reg 0.01

python learner.py --cuda --dataset yago15k --model TPComplEx --rank 1892 --no_time_emb --emb-reg-type N3 --emb-reg 0.1 --time-reg-type Lambda3 --time-reg 1

python learner.py --cuda --dataset gdelt --model TPComplEx --rank 1256 --emb-reg-type N3 --emb-reg 1e-4 --time-reg-type Lambda3 --time-reg 1e-2
```

**TNTComplEx**

```sh
python learner.py --cuda --dataset ICEWS14 --model TNTComplEx --rank 156 --emb-reg-type N3 --emb-reg 1e-2 --time-reg-type Linear3 --time-reg 1e-2

python learner.py --cuda --dataset ICEWS05-15 --model TNTComplEx --rank 128 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Linear3 --time-reg 1

python learner.py --cuda --dataset yago15k --model TNTComplEx --rank 189 --emb-reg-type N3 --no_time_emb --emb-reg 1e-2 --time-reg-type Linear3 --time-reg 1
```

## Acknowledgement

We refer to the code of [TComplEx](https://github.com/facebookresearch/tkbc) and [TPComplEx](https://github.com/Jinfa/TPComplEx). Thanks for their contributions.

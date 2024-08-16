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

```
python learner.py --cuda --dataset ICEWS14 --model TPComplEx --rank 1594 --emb-reg-type N3 --emb-reg 1e-2 --time-reg-type Lambda3 --time-reg 1e-2

python learner.py --cuda --dataset ICEWS05-15 --model TPComplEx --rank 886 --emb-reg-type N3 --emb-reg 0.01 --time-reg-type Lambda3 --time-reg 0.01

python learner.py --cuda --dataset yago15k --model TPComplEx --rank 1892 --no_time_emb --emb-reg-type N3 --emb-reg 0.1 --time-reg-type Lambda3 --time-reg 1

python learner.py --cuda --dataset gdelt --model TPComplEx --rank 1256 --emb-reg-type N3 --emb-reg 1e-4 --time-reg-type Lambda3 --time-reg 1e-2
```

**TLT_KGE_Complex** & **TLT_KGE_Quaternion**

```
python learner.py --dataset ICEWS14 --model TLT_KGE_Quaternion --rank 1200 --emb-reg-type N3 --emb-reg 3e-3 --time-reg-type Lambda32 --time-reg 3e-2 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --cuda

python learner.py --dataset ICEWS14 --model TLT_KGE_Complex --rank 1200 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Lambda32 --time-reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --cuda

python learner.py --dataset ICEWS05-15 --model TLT_KGE_Quaternion --rank 1200 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Lambda32 --time-reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 1440 --cuda

python learner.py --dataset ICEWS05-15 --model TLT_KGE_Complex --rank 1200 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Lambda32 --time-reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 1440 --cuda

python learner.py --dataset gdelt --model TLT_KGE_Quaternion --rank 1500 --emb-reg-type N3 --emb-reg 5e-4 --time-reg-type Lambda32 --time-reg 3e-2 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --cuda

python learner.py --dataset gdelt --model TLT_KGE_Quaternion --rank 1500 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Lambda32 --time-reg 1e-1 --valid_freq 5 --max_epoch 200 --learning_rate 0.1 --batch_size 1000  --cycle 120 --cuda
```

**TeAST**

```
python  learner.py --cuda --model TeAST --dataset ICEWS14 --emb-reg-type N3 --emb-reg 0.0025 --time-reg-type Spiral3 --time-reg 0.01

python  learner.py --cuda --model TeAST --dataset ICEWS05-15 --emb-reg-type N3 --emb-reg 0.002 --time-reg-type Spiral3 --time-reg 0.1

python  learner.py --cuda --model GDELT  --dataset ICEWS05-15 --emb-reg-type N3 --emb-reg 0.003 --time-reg-type Spiral3 --time-reg 0.003
```

**QDN**

```
python learner.py --dataset ICEWS14 --model QDN --rank 2000 --emb-reg-type N3 --emb-reg 0.0075 --time-reg-type Linear3 --time-reg 0.01 --cuda

python learner.py --dataset ICEWS05-15 --model QDN --rank 2000 --emb-reg-type N3 --emb-reg 0.0025 --time-reg-type Linear3 --time-reg 0.1 --cuda

python learner.py --dataset yago11k --model QDN --rank 2000 --emb-reg-type N3 --emb-reg 0.025 --time-reg-type Linear3 --time-reg 0.001 --cuda

python learner.py --dataset wikidata12k --model QDN --rank 2000 --emb-reg-type N3 --emb-reg 0.025 --time-reg-type Linear3 --time-reg 0.0025 --cuda

```

**TeLM**

```
python learner.py --dataset ICEWS14 --model TeLM --rank 2000 --emb-reg-type N3 --emb-reg 0.0075 --time-reg-type Linear3 --time-reg 0.01 --cuda

python learner.py --dataset ICEWS05-15 --model TeLM --rank 2000 --emb-reg-type N3 --emb-reg 0.0025 --time-reg-type Linear3 --time-reg 0.1 --cuda

python learner.py --dataset yago11k --model TeLM --rank 2000 --emb-reg-type N3 --emb-reg 0.025 --time-reg-type Linear3 --time-reg 0.001 --cuda

python learner.py --dataset wikidata12k --model TeLM --rank 2000 --emb-reg-type N3 --emb-reg 0.025 --time-reg-type Linear3 --time-reg 0.0025 --cuda

```

**TNTComplEx**

```
python learner.py --cuda --dataset ICEWS14 --model TNTComplEx --rank 156 --emb-reg-type N3 --emb-reg 1e-2 --time-reg-type Linear3 --time-reg 1e-2

python learner.py --cuda --dataset ICEWS05-15 --model TNTComplEx --rank 128 --emb-reg-type N3 --emb-reg 1e-3 --time-reg-type Linear3 --time-reg 1

python learner.py --cuda --dataset yago15k --model TNTComplEx --rank 189 --emb-reg-type N3 --no_time_emb --emb-reg 1e-2 --time-reg-type Linear3 --time-reg 1
```

## Acknowledgement

We refer to the code of [TComplEx](https://github.com/facebookresearch/tkbc). Thanks for their contributions.

| Dataset     | Model | Rank | Emb Reg | Emb Reger | Time Reg | Time Reger |
| ----------- | ----- | ---- | ------- | --------- | -------- | ---------- |
| ICEWS14     | QDN   | 2000 | 0.0075  | N3        | 0.01     | Lambda3    |
| ICEWS05-15  | QDN   | 2000 | 0.0025  | N3        | 0.1      | Lambda3    |
| yago11k     | QDN   | 2000 | 0.025   | N3        | 0.001    | Lambda3    |
| wikidata12k | QDN   | 2000 | 0.025   | N3        | 0.0025   | Lambda3    |
| ICEWS14     | TeAST | 2000 | 0.0025  | N3        | 0.01     | Lambda3    |
| ICEWS05-15  | TeAST | 2000 | 0.002   | N3        | 0.1      | Lambda3    |
| GDELT       | TeAST | 2000 | 0.003   | N3        | 0.003    | Lambda3    |
| ICEWS14     | TeLM  | 2000 | 0.0075  | N3        | 0.01     | Lambda3    |
| ICEWS05-15  | TeLM  | 2000 | 0.0025  | N3        | 0.1      | Lambda3    |
| yago11k     | TeLM  | 2000 | 0.025   | N3        | 0.001    | Lambda3    |
| wikidata12k | TeLM  | 2000 | 0.025   | N3        | 0.0025   | Lambda3    |

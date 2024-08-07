# DeepTune: Eﬃcient Layer-wise Feature Extraction and Fine-Tuning of Foundation Models


## Installation

**Conda Environment**

To create a conda environment with the necessary dependencies, run the following commands:

```bash
git clone https://github.com/abwerby/DeepTune.git
conda env create -f env.yaml
conda activate deeptune
```

## Datasets

We provide a set of datasets that can be used to evaluate the performance of DeepTune.
The datset will be downloaded automatically when running the pipeline for the first time.

The downloaded datasets will have the following structure:
```bash
./data
├── fashion
│   ├── images_test
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── images_train
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   ├── description.md
│   ├── fashion.tgz
│   ├── test.csv
│   └── train.csv
├── emotions
    ...
...
```
Feel free to explore the images and the `description.md` files to get a better understanding of the datasets.
The following table will provide you an overview of their characteristics and also a reference value for the 
accuracy that a naive AutoML system could achieve on these datasets:

| Dataset name | # Classes | # Train samples | # Test samples | # Channels | Resolution | Reference Accuracy |
|--------------|-----------|-----------------|----------------|------------|------------|--------------------|
| fashion      | 10        | 60,000          | 10,000         | 1          | 28x28      | 0.88               |
| flowers      | 102*      | 5732            | 2,457          | 3          | 512x512    | 0.55               |
| emotions     | 7         | 28709           | 7,178          | 1          | 48x48      | 0.40               |
| skin_cancer  | 7*        | 7,010           | 3,005          | 3          | 450x450    | 0.71               |

*classes are imbalanced

## Run DeepTune on sample dataset

To search for the best configuration on the sample Dataset, run the `tune.py` script.
to change the configuration for the search, change the `deep_tune_config.yaml` file in the `config` directory.

```bash
python tune.py
```

This will save alot of configurations in the a directory with the name `XxxxxxxDataset-xxxxx` in the main directory, 
the file structure will look like this:

```
.
├── XxxxxxxDataset-xxxxx
│   ├── 0
│   │   ├── config.json
│   │   ├── model.pth
│   │   ├── std.out
│   │   └── std.err
│   ├── 1
│   │   ├── config.json
│   │   ├── model.pth
│   │   ├── std.out
│   │   └── std.err
│   
```

to get the predictions for the best configuration found by the search, run the `test.py` script.
change the `test_config.yaml` file in the `config` directory to the configuration number found by the search.
then run the following command:

```bash
python test.py 
```

This will generate the `predictions.npy` for the found configuration and save it in the `data/exam_dataset` directory.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
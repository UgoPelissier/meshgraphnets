# Learning Mesh-Based Simulation with Graph Networks

## Setup
```bash
conda env create -f utils/envs/gnn.yml
conda activate gnn
```

## Download the dataset
```bash
bash download_dataset.sh cylinder_flow data/
```

It will create a folder `cylinder_flow/` inside the folder `data/` and will take some time to download the simulations. You will obtain four files:
- `meta.json`
- `test.tfrecord`
- `train.tfrecord`
- `valid.tfrecord`

Rename the folder `cylinder_flow/` to `raw/`.

## Parameters
Open `main.py` to update the paths.

The dataset is composed of 1000 simulations, each of them having 600 time steps. Pre-processing and training on all the dataset is not possible on standards GPU in a short time. To select only a part of the dataset, have a look at the `data.*` parameters. These are already set up, but you can change to reduce the number of data and have a faster training.

## Train the model
Train the model by running:
```bash
python main.py fit
```

You can get help on the command line arguments by running:
```bash
python main.py fit --help
```

It will create a new folder in the `logs/` folder containing the checkpoints of the model and a configuration file containing the parameters used for the training, that you can use later if you want.

## Evaluate the model
To evaluate the model training, run:
```bash
tensorboard --logdir=logs/
```

You can stop the training whenever you are satisfied with the learning. The model is saved in `logs\version_*\checkpoints\`.

## Test the model
To test the model, run:
```bash
python main.py test --ckpt_path $ckpt_path
```
where `$ckpt_path` is the path to the checkpoint file located in the `logs/version_$version/checkpoints/` folder.

## Contact

Ugo Pelissier \
\<[ugo.pelissier@minesparis.psl.eu](mailto:ugo.pelissier@eminesparis.psl.eu)\>

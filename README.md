# KERL

[Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems](https://arxiv.org/abs/2312.10967)

<img src="assets/model_overview_v2-1.png" width = "1000" />

KERL leverages the power of knowledge graphs and pre-trained language models to generate semantically rich entity representations, enabling more accurate recommendations and engaging user interactions.

## Installation Instructions

To run the provided code, follow these steps:

1. **Download and install Miniconda**:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Create a new conda environment with provided `environment.yml` file**:

   ```bash
   conda env create -f environment.yml
   conda activate kerl
   ```

> :information_source: **Note:** 
> 1. Code tested only on WSL2 and Linux-based systems.
> 2. The provided Miniconda installation commands are for Linux. For other systems, download the appropriate installer from the [official website](https://docs.conda.io/en/latest/miniconda.html).

## Quick-Start


To train the KERL model, use the following commands. Select the appropriate configuration file based on the dataset:

```bash
# For training on the Inspired dataset
python main.py -c config/inspired_kerl.yaml

# For training on the ReDial dataset
python main.py -c config/redial_kerl.yaml

```

> :information_source: **Note:** You will need a wandb account to log the training metrics. see [here](https://docs.wandb.ai/quickstart) for more details.


## Saved Models

You can download the saved models for two datasets from the following links:

1. Inspired Model: [Download](https://www.dropbox.com/scl/fo/urnpek5rcq2vantxpxwkt/h?rlkey=tte9f6a5nnjii6zd79igvwt27&dl=0)
2. Redial Model: [Download](https://www.dropbox.com/scl/fo/pmb12w3fnfu2k0t6tsrav/h?rlkey=gy59emftp8nd4mnyaqk8wyn2l&dl=0)


Place the downloaded `saved` folder in the root directory.
In the configuration file for the respective dataset, set `<phase>_reload` to `True`,
and `<phase>_model_path` with the path of the downloaded model.


## Bibinfo

```
@article{qiu_kerl_2023,
	author       = {Zhangchi Qiu and Ye Tao and Shirui Pan and Alan Wee-Chung Liew},
	title        = {Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems},
	year         = 2023,
	journal      = {ArXiv},
	url          = {https://arxiv.org/abs/2312.10967}
```
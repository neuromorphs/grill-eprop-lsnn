# grill-eprop-lsnn Phone Classifyer

In this subproject we will try to compute the initialisation weights of the the e-prop based LSNN <sup id="a-Voelker-2019">[1, ](#f-Voelker-2019)</sup> designed to classify phonemes, using the Legendre Memory Unit <sup id="a-bellec-2020">[2](#f-bellec-2020)</sup>.


## Introduction

This project is inspired by the discussion started in the grill workgroup [Telluride 2020 Neuromorphic Engineering Workshop](https://sites.google.com/view/telluride2020/home). 
The target will be the investigation of possible alternatives for coupling LMU and e-prop methods and generate heterogeneus LSNNs which will be evaluated on the classification performances of the TIMT dataset.



## Getting Started

At the moment the repo is organised to work locally so we use the below instructions to setup our development environment and to download and set-up the dataset. 

### Python Environment (for CPU)

* Install [miniconda](https://docs.conda.io/en/latest/miniconda.html). After miniconda is installed, open a new Terminal or "Anaconda Prompt".
* Create a new conda environment with all the dependencies: 
```
conda create --name lmu-e-prop --file requirements_conda.txt
conda activate lmu-e-prop
pip install -r requirements_pip.txt
```
### Python Environment (for GPU)

* Install [miniconda](https://docs.conda.io/en/latest/miniconda.html). After miniconda is installed, open a new Terminal or "Anaconda Prompt".
* Create a new conda environment with all the dependencies: 
```
conda create --name lmu-e-prop-gpu --file requirements_conda_gpu.txt
conda activate lmu-e-prop-gpu
pip install -r requirements_pip_gpu.txt
```
* The tensorflow-gpu module is only compatible with CUDA version 10.0. If you have multiple versions of CUDA installed, you may need to add the CUDA 10.0 libraries to your paths. For example:
```
LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
```
### Basic steps

* Activate nengo jupyter extension: `jupyter serverextension enable nengo_gui.jupyter`
* Clone this repository: `git clone https://github.com/neuromorphs/grill-eprop-lsnn && cd grill-eprop-lsnn/pc`
* Process the dataset by running: `python3 timit_processing.py`
* Train an LSTM with symmetric e-prop: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_framewise_lstm.py --eprop=symmetric --preproc=mfccs`
* Train an LSNN with symmetric e-prop: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_framewise_lsnn.py --eprop=symmetric --preproc=mfccs`
* If you have multiple GPU vrite the correct id in the variable CUDA_VISIBLE_DEVICES (`CUDA_VISIBLE_DEVICES=0,1` for using the first two GPUs)
* TODO Add the next steps of the project



### Data and Preprocess

TIMT data 
is included for convenience in the ./timt folder original data is available 
can be downloaded from three different sources:
* The original website proposed in the e-prop paper [here](https://catalog.ldc.upenn.edu/LDC93S1).
* An alternative link with a compressed file is [here](https://figshare.com/articles/TIMIT_zip/5802597). 
* The Kaggle version providing all the files and some notebooks for exploring the dataset [here](https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech) that require a simple registration (using google account). 

Downloaded files must be adapted with te bash script `rename-files.sh` and moved in a falder called timit. 
Operations to set-up the dataset are here proposed as an extract of the procedure defined in the [original e-prop repository](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_2_TIMIT/README.md)

The `timit_processing.py` [script](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_2_TIMIT/timit_processing.py) will be used to process the raw TIMIT dataset to features appropriate for training the models.
Inside the `timit_processing.py` script you should change the `DATA_PATH` variable to point to the source data,
which should be the TIMIT dataset (with or without additional HTK feature files). Start the preprocessing with:

    python3 timit_processing.py


A more involved alternative is using the `HTK` software package instructions can be found [here](https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_2_TIMIT/README.md).



### Examples to be used (Preliminary)

In the following the examples we will use in the project:
* [How to extract connection weights from nengo models](https://github.com/neuromorphs/grill-lmu/blob/master/weights/Connection%20Weights%20in%20Nengo.ipynb) to take inspiration abut the extraction of weights from the nengo ensamble connection.
* [A possible solution to iteratively investigate several parameter configuration using pytry](https://github.com/neuromorphs/grill-eprop-lsnn/blob/master/rl/Using%20pytry%20to%20explore%20parameters.ipynb)
* [The LMU-based network for the classification of psMNIST dataset](https://www.nengo.ai/nengo-dl/examples/lmu.html#)
* [The e-prop LSTM and LSNN networks for the classification of the TIMIT dataset ](https://github.com/IGITUGraz/eligibility_propagation/tree/master/Figure_2_TIMIT)


Many of these examples uses [Nengo](https://www.nengo.ai/) to build and parameterize the model.
We have also installed the packages useful for the automatic exploration of the parameter configuration [nni](https://github.com/Microsoft/nni) and [pytry](https://github.com/tcstewar/pytry) 

## Methods

### Action plan

### Constraints

### Evaluating the Model

## Footnotes and references

* <a id="f-Voelker-2019" href="#a-Voelker-2019"><sup>1</sup></a> [Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks. Voelker et al., Advances in Neural Information Processing Systems, 2019](https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf)
* <a id="f-bellec-2020" href="#a-bellec-2020"><sup>2</sup></a> [A solution to the learning dilemma for recurrent networks of spiking neurons. Bellec et al., Nature Communications, 2020](https://www.jneurosci.org/content/31/17/6266.short)

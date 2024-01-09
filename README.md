# E/LRA

A streamlined variant of [LRA](https://github.com/google-research/long-range-arena/) with

- [pinned dependencies](https://github.com/lucaslingle/e-lra/blob/main/setup.py#L19-L48), avoiding installation difficulties; 
- [helpful examples](https://github.com/lucaslingle/e-lra/tree/main?tab=readme-ov-file#usage), improving productivity;
- [automatic dataset setup](https://github.com/lucaslingle/e-lra/blob/main/prep_data.sh), avoiding manual wrangling;
- [complete factory function](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/utils/train_utils.py#L35-L128), supporting all the models implemented by LRA;
- [well-defined variables](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/image/input_pipeline.py#L21), avoiding crashing scripts;
- [deterministic shuffling](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/image/input_pipeline.py#L52-59), aiding reproducibility. 

## Installation

It is recommended to install the dependencies using a virtual environment such as venv or miniconda.
After the virtual environment is activated, run: 
```
pip3 install --upgrade pip;
git clone https://github.com/lucaslingle/e-lra.git;
cd e-lra;

##### CPU-Only #####
pip3 install -e '.[cpu]' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html;

##### Nvidia GPU, CUDA 11 #####
pip3 install -e '.[cuda11]' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;

##### Cloud TPU VM #####
pip3 install -e '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```

## Prepare Data

To prepare the data, run:
```
source ./prep_data.sh;
```

## Examples

To benchmark a vanilla transformer on all tasks, run the following.  
To benchmark a different xformer, change the ```config``` option. 

Test results are printed and written to a file ```results.json``` in the specified ```model_dir```. 

### ListOps
```
python3 lra_benchmarks/listops/train.py \
      --task_name=basic \
      --config=lra_benchmarks/listops/configs/transformer_base.py \
      --config.eval_frequency=1000 \
      --data_dir=lra_data/listops/ \
      --model_dir=/tmp/listops/;
```

### Text Classification
Sweep over ```MAX_LENGTH=1000,2000,3000,4000```, and report the best result.
```
python3 lra_benchmarks/text_classification/train.py \
      --task_name=imdb_reviews \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --config.eval_frequency=1000 \
      --config.max_length=$MAX_LENGTH \
      --data_dir=lra_data/text_classification/ \
      --model_dir=/tmp/text_classification/;

# Clean up model_dir after viewing test metrics,
# since we need to run from scratch for each MAX_LENGTH setting!
rm -rf /tmp/text_classification/;
```

### Retrieval
```
python3 lra_benchmarks/retrieval/train.py \
      --task_name=basic \
      --config=lra_benchmarks/retrieval/configs/transformer_base.py \
      --config.eval_frequency=1000 \
      --data_dir=lra_data/retrieval/ \
      --model_dir=/tmp/retrieval/;
```

### Image Classification
```
python3 lra_benchmarks/image/train.py \
      --task_name=cifar10 \
      --config=lra_benchmarks/image/configs/cifar10/transformer_base.py \
      --config.eval_frequency=1000 \
      --model_dir=/tmp/image/;
```

### Pathfinder
```
python3 lra_benchmarks/image/train.py \
      --task_name=pathfinder32_easy \
      --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
      --config.eval_frequency=1000 \
      --model_dir=/tmp/pathfinder/;
```

### Acknowledgements

Experiments supported by Cloud TPUs from Google's TPU Research Cloud (TRC).

# E/LRA

A working fork of [LRA](https://github.com/google-research/long-range-arena/) with 
- [pinned dependencies](https://github.com/lucaslingle/e-lra/blob/main/setup.py#L19-L48), avoiding installation difficulties; 
- [helpful examples](https://github.com/lucaslingle/e-lra/tree/main?tab=readme-ov-file#usage), improving productivity;
- [automatic dataset setup](https://github.com/lucaslingle/e-lra/blob/main/prep_data.sh), avoiding manual wrangling;
- [complete factory function](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/utils/train_utils.py#L35-L128), supporting all the models implemented by LRA; 
- [task-consistent flags](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/image/train.py#L51), simplifying test-time evaluation;  
- [defined data path variables](https://github.com/lucaslingle/e-lra/blob/main/lra_benchmarks/image/input_pipeline.py#L21), avoiding crashing scripts.

The changes are non-invasive to the original source code, but significantly streamline usage of the LRA task suite. 

## Installation

It is recommended to install the python dependencies using a virtual environment such as venv, pipenv, or miniconda.
After the virtual environment is activated, run: 
```
pip3 install --upgrade pip;
git clone https://github.com/lucaslingle/e-lra.git;
cd e-lra;

##### CPU-Only #####
pip3 install '.[cpu]' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html;
    
##### Nvidia GPU, CUDA 11 #####
pip3 install '.[cuda11]' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;

##### Cloud TPU VM #####
pip3 install '.[tpu]' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
```
To install in editable mode, write ```pip3 install -e ...```.  
To install via pipenv, write ```pipenv install ...```. 

## Prepare Data

To prepare the data, run:
```
source ./prep_data.sh;
```

## Examples

To benchmark a vanilla transformer on all tasks, run the following.  
To benchmark a different xformer, change the ```config``` option. 

Test metrics are written to a file ```results.json``` in the specified ```--model_dir```. 

### ListOps
```
python3 lra_benchmarks/listops/train.py \
      --config=lra_benchmarks/listops/configs/transformer_base.py \
      --model_dir=/tmp/listops \
      --task_name=basic \
      --data_dir=lra_data/listops/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100;

python3 lra_benchmarks/listops/train.py \
      --config=lra_benchmarks/listops/configs/transformer_base.py \
      --model_dir=/tmp/listops \
      --task_name=basic \
      --data_dir=lra_data/listops/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --test_only=True;
```

### Text Classification
Sweep over max_length=1000,2000,3000,4000, report the best result.
```
python3 lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --model_dir=/tmp/text_classification/ \
      --task_name=imdb_reviews \
      --data_dir=lra_data/text_classification/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --config.max_length=1000;  

python3 lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --model_dir=/tmp/text_classification/ \
      --task_name=imdb_reviews \
      --data_dir=lra_data/text_classification/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --config.max_length=1000 \
      --test_only=True;

# Here we clean up model_dir after viewing test metrics,
# since we need to run from scratch for each max_length setting!
rm -rf /tmp/text_classification/;
```

### Retrieval
```
python3 lra_benchmarks/retrieval/train.py \
      --config=lra_benchmarks/retrieval/configs/transformer_base.py \
      --model_dir=/tmp/retrieval \
      --task_name=basic \
      --data_dir=lra_data/retrieval/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100;

python3 lra_benchmarks/retrieval/train.py \
      --config=lra_benchmarks/retrieval/configs/transformer_base.py \
      --model_dir=/tmp/retrieval \
      --task_name=basic \
      --data_dir=lra_data/retrieval/ \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --test_only=True;
```

### Image Classification
```
python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/cifar10/transformer_base.py \
      --model_dir=/tmp/image/ \
      --task_name=cifar10 \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100;

python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/cifar10/transformer_base.py \
      --model_dir=/tmp/image/ \
      --task_name=cifar10 \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --test_only=True;
```

### Pathfinder
```
python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
      --model_dir=/tmp/pathfinder/ \
      --task_name=pathfinder32_hard \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100;

python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/pathfinder32/transformer_base.py \
      --model_dir=/tmp/pathfinder/ \
      --task_name=pathfinder32_hard \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --test_only=True;
```

### Path-X
```
python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/pathfinder128/transformer_base.py \
      --model_dir=/tmp/pathx/ \
      --task_name=pathfinder128_hard \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100;

python3 lra_benchmarks/image/train.py \
      --config=lra_benchmarks/image/configs/pathfinder128/transformer_base.py \
      --model_dir=/tmp/pathx/ \
      --task_name=pathfinder128_hard \
      --config.checkpoint_freq=100 \
      --config.eval_frequency=100 \
      --test_only=True;
```

#### Note
The default config for vanilla transformer does not work with Path-X on TPU v3 due to OOM and large batch size. No configs were provided for Path-X for any other model.

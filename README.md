   
# DISCOVER

DISsecting COmpositionality in VEctor Representations



## Requirements
0. Create a python venv to run code in:
```
python -m venv .venv
```

1. Activate the venv
```
source .venv/bin/activate
```

2. Install requirements
```
pip install -U pip setuptools wheel
pip install torch
pip install numpy

pip install transformers
pip install accelerate
```


# Produce datasets

One of the functionalities of this repository is producing datasets that can serve as inputs to LLMs in order to analyze the representations they produce when fed these inputs. The following code shows how to do that; it can be applied in any of the six directories corresponding to different data domains (namely, passive/, tense/, subjaux/, logic_1hop/, or arithmetic/)

```
# This directory could alternatively be arithmetic/,
# logic_1hop/, passive/, tense/, or subjaux/
cd code_function
mkdir base_sentences

# Run this only if you're in passive/ or tense/
cp ../subjaux/cfg_gen.py .

python dataset_generation.py 
python intervention_generation_content.py
```

# Run interventions

The other functionality of this codebase is for performing causal interventions on LLM representations. Here we provide instructions for doing this. Note that these experiments are computationally intensive: They require GPUs, and the second and third commands take one to two days each.

1. Copy in other code that will be necessary
```
git clone https://github.com/tommccoy1/discover.git
cp interventions_ensemble.py discover/
cd discover/
```

2. Analyze the LLM with DISCOVER
```
python decompose_ensemble.py --llm_name gpt-oss --discover_role_scheme passive_bi --discover_emb_size 100 --discover_role_emb_size 50 --log_dir passive/logs/ --weight_dir passive/weights/ --data_dir passive/data/ --truncate_after_strs '<|end|>,<' --truncate_before_strs 'final<|message|>' --delete_strings ")}" --batch_size 24 --prefix_filename prefix.txt --lr 0.0001 --patience 20 --learning_rate_decay_patience 10 --eval_every 200 --eval_acc_every 2000 --valid_stop_early_index 100 --valid_acc_stop_early_index 100 --max_length 50
```

3. Perform causal interventions
```
python interventions_ensemble.py --json_args passive/logs/tpe_ensemble_gpt-oss_passive_bi_0.decomposition_args.json --random_seed 12345 --intervention_type content --intervention_all_bidirectional --log_suffix _0
```



 

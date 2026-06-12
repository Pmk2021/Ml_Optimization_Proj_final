# Ml_Optimization_Proj_final

## Installation

To install all required packages, first create and activate a Conda environment named `optimizer_project`, then install the dependencies:

```bash
# Create the conda environment
conda create -n optimizer_project python=3.11 -y

# Activate the environment
conda activate optimizer_project

# Install required packages
pip install transformers datasets torch accelerate safetensors
```

Then, download the supra-50-m base model from here: https://huggingface.co/SupraLabs/Supra-50M-Base.

Ensure that the files `config.json`, `model.safetensors`, and `tokenizer.json` are in a folder called `base_model`. You may also use any other huggingface style LLM, as long as it includes the above files.

Finally, download the finetuning dataset from here: https://huggingface.co/datasets/yahma/alpaca-cleaned. This is the dataset each finetuning script will train your base model on. Ensure the file is saved as `alpaca_data_cleaned.json` in the base directory

## Finetuning LLMs
 There are 6 main fine-tuning training scripts in the folder `scripts/`:

 1) `finetune.py`: Finetunes entire model
 2) `finetune_lora.py`: Finetunes model with Lora(traines about 2% of the model)
 3) `finetune_smallest.py`: Finetunes only smallest weights by magnitude
 4) `finetune_smallest_grad.py`: Finetuned only smallest weights by gradient magnitude
 5) `finetune_biggest.py`: Finetunes only biggest weights by magnitude
 6) `finetune_biggest_grad.py`: Finetuned only biggest weights by gradient magnitude

 Due to the long training time, each of these files only trains a single model. They can be called as follows. 
 
 Scripts `finetune.py` and `finetune_lora.py` are simply called as:

 ```bash
python finetune.py
 ```

 or 

 ```bash
 python finetune_lora.py
 ```

 The remaining scripts must have the percent of weights that you want to train passed in. This can be done with the `--density` flag as such:

 ```bash
 python finetune_biggest_grad.py density 0.1
 ```

 The above script will only train the top 10% of weights by gradient size every timestep.

 ## Evaluation

 Evaluations can be conducted using the `EleutherAI lm-evaluation-harness`(lm-evaluation-harness).

 You may clone the repository into this folder, and follow their setup instructions. To evaluate a finetuned model on the benchmark using the same settings as in the report, run the following command:


 ```
 lm_eval --model hf  --model_args pretrained=[path_to_model] --tasks [task_name] --device cuda:0   --batch_size 4 --limit 1000 --num_fewshot 2
 ```

Where `[task_name]` is one of the following:

1) xwinograd_en
2) mmlu_social_sciences
3) babi 
4) triviaqa
5) Piqa

And `[path_to_model]` is the path to the folder containing files to the finetuned model you wish to evaluate.

Due to time/compute constraints, each of these commands were run one at a time on a laptop, though they could possibly be parallelized on a more powerful computer.
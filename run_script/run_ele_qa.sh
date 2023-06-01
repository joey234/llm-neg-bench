export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="neqa.csv" #@param {"type": "string"}
# model_names=("ada" "babbage" "curie" "davinci") #@param {"type": "raw"}
export model_names_string="gpt-neo-125M gpt-neo-1.3B gpt-neo-2.7B gpt-j-6B"

rm -rf results/neqa/eleutheur

CUDA_VISIBLE_DEVICES=1 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir neqa/loss/eleutheur \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

CUDA_VISIBLE_DEVICES=1 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir neqa/acc/eleutheur \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu


# we have to use %run because executing with !python does not load the python code in the colab shell
# /content/inverse-scaling-eval-pipeline/eval_pipeline/plot_loss.py \
#   /content/results \
#   --task-type $evaluation_metric
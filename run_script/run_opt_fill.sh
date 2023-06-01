#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="mwr_prompt.csv" #@param {"type": "string"}
# model_names=("ada" "babbage" "curie" "davinci") #@param {"type": "raw"}
# export model_names_string="opt-125m opt-350m opt-1.3b opt-2.7b"
export model_names_string="opt-1.3b"

CUDA_VISIBLE_DEVICES=1 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir results \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu

# we have to use %run because executing with !python does not load the python code in the colab shell
# /content/inverse-scaling-eval-pipeline/eval_pipeline/plot_loss.py \
#   /content/results \
#   --task-type $evaluation_metric
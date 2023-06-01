export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="negnli_neg_prompts.csv" #@param {"type": "string"}
# model_names=("ada" "babbage" "curie" "davinci") #@param {"type": "raw"}
export model_names_string="flan-t5-small flan-t5-base flan-t5-large flan-t5-xl flan-t5-xxl"
# export model_names_string="gpt-j-6B"

export exp_dir="negnli/neg_prompts/flant5"

rm -rf results/$exp_dir

# CUDA_VISIBLE_DEVICES=0 python eval_pipeline/main.py \
# 	--dataset-path "$file_name" \
# 	--exp-dir $exp_dir/loss \
# 	--models $model_names_string \
# 	--task-type $evaluation_metric \
# 	--use-gpu


export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

CUDA_VISIBLE_DEVICES=0 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir $exp_dir/acc \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu

# export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
# python eval_pipeline/plot_loss.py \
#   $exp_dir/loss \
#   --task-type $evaluation_metric

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric
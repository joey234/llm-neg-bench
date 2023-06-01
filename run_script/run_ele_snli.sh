export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="snli_flan_prompts.csv" #@param {"type": "string"}
# export model_names_string="opt-125m opt-350m opt-1.3b opt-2.7b"
export model_names_string="gpt-neo-125M gpt-neo-1.3B gpt-neo-2.7B gpt-j-6B"
# export model_names_string="opt-13b"
export exp_dir="snli/flan_prompts/eleutheur"

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


# python convert_results.py -i results/$expr_dir/gpt-neo-125M.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-1.3B.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-2.7B.csv
# python convert_results.py -i results/$expr_dir/gpt-j-6B.csv

# export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
# python eval_pipeline/plot_loss.py \
#   $exp_dir/loss \
#   --task-type $evaluation_metric

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric
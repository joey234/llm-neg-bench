export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="mkrnq_t5_pos_prompt.csv" #@param {"type": "string"}
# model_names=("ada" "babbage" "curie" "davinci") #@param {"type": "raw"}
export model_names_string="flan-t5-small flan-t5-base flan-t5-large flan-t5-xl"
# export model_names_string="gpt-j-6B"

export exp_dir="mkr_test/pos_prompts/flant5/contrastive"

# rm -rf results/$exp_dir

# CUDA_VISIBLE_DEVICES=1 python eval_pipeline/main.py \
# 	--dataset-path "$file_name" \
# 	--exp-dir $exp_dir \
# 	--models $model_names_string \
# 	--task-type $evaluation_metric \
# 	--use-gpu


# python convert_results.py -i results/$expr_dir/gpt-neo-125M.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-1.3B.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-2.7B.csv
# python convert_results.py -i results/$expr_dir/gpt-j-6B.csv

python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric
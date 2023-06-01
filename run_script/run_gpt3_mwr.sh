# export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
# export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="mwr_prompt.csv" #@param {"type": "string"}
# export model_names_string="text-ada-001 text-babbage-001 text-curie-001 text-davinci-003"
export model_names_string="text-davinci-001"

export exp_dir="mwr_test/gpt3-001"

rm -rf results/$exp_dir

python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir $exp_dir \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu \
	--contrastive-search True \
	--batch-size 100

python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric

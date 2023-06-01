export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
export file_name="sar_test_prompt_ins.csv" #@param {"type": "string"}
# export model_names_string="text-ada-001 text-babbage-001 text-curie-001 text-davinci-003"
export model_names_string="text-davinci-003"

export exp_dir="sar/neg_prompts/gpt3"

rm -rf results/$exp_dir

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

CUDA_VISIBLE_DEVICES=0 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir $exp_dir/acc \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--batch-size 100



export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric
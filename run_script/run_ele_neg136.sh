export HF_DATASETS_CACHE="/data/projects/punim0478/thinh/hf_cache"
export TRANSFORMERS_CACHE="/data/projects/punim0478/thinh/hf_cache"

#@title Running GPT-3 and plotting the results { display-mode: "form" }
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
# export file_name="mkrnq_prompt.csv" #@param {"type": "string"}
# model_names=("ada" "babbage" "curie" "davinci") #@param {"type": "raw"}
# export model_names_string="gpt-neo-125M gpt-neo-1.3B gpt-neo-2.7B"
export model_names_string="gpt-j-6B"

export exp_dir="neg136_test/eleutheur/constrastive"
# rm -rf results/$exp_dir

export file_name="neg-136-nat-onlyneg.csv" #@param {"type": "string"}
CUDA_VISIBLE_DEVICES=0 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir $exp_dir/nat \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu

export file_name="neg-136-simp-onlyneg.csv" #@param {"type": "string"}

CUDA_VISIBLE_DEVICES=0 python eval_pipeline/main.py \
	--dataset-path "$file_name" \
	--exp-dir $exp_dir/simp \
	--models $model_names_string \
	--task-type $evaluation_metric \
	--use-gpu

# python convert_results.py -i results/$expr_dir/gpt-neo-125M.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-1.3B.csv
# python convert_results.py -i results/$expr_dir/gpt-neo-2.7B.csv
# python convert_results.py -i results/$expr_dir/gpt-j-6B.csv

export exp_dir_ori="neg136_test/eleutheur/constrastive"


export exp_dir=$exp_dir_ori/nat
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir=$exp_dir_ori/simp
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric
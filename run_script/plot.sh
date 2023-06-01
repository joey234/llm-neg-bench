#MWR
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

export exp_dir="mwr/eleutheur/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir="mwr/gpt2/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric

export exp_dir="mwr/opt/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


#MKR
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

export exp_dir="mkr/eleutheur/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir="mkr/gpt2/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric

export exp_dir="mkr/opt/contrastive"
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric

#NAN

export exp_dir="nan/eleutheur/contrastive"
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/loss \
  --task-type $evaluation_metric

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric

export exp_dir="nan/gpt2/contrastive"
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/loss \
  --task-type $evaluation_metric

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric


export exp_dir="nan/opt/contrastive"
export evaluation_metric="classification" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/loss \
  --task-type $evaluation_metric

export evaluation_metric="classification_acc" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]
python eval_pipeline/plot_loss.py \
  $exp_dir/acc \
  --task-type $evaluation_metric


#NEG136
export exp_dir_ori="neg136/eleutheur/constrastive"
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

export exp_dir=$exp_dir_ori/nat
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir=$exp_dir_ori/simp
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric



export exp_dir_ori="neg136/gpt2/constrastive"
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

export exp_dir=$exp_dir_ori/nat
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir=$exp_dir_ori/simp
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir_ori="neg136/opt/constrastive"
export evaluation_metric="hitrate" #@param ["classification", "sequence_prob", "logodds", "absolute_logodds", "classification_acc"]

export exp_dir=$exp_dir_ori/nat
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric


export exp_dir=$exp_dir_ori/simp
python eval_pipeline/plot_loss.py \
  $exp_dir \
  --task-type $evaluation_metric
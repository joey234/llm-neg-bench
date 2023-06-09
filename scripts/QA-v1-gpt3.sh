SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=QA-v1-gpt3-new
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset QA_bias-v1 \
    --task-type logodds \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 20 \
&&
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type logodds \
    $EXP_DIR



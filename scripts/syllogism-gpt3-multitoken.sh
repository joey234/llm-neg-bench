SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=syllogism-gpt3-multitoken
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset syllogism-0shot-multitoken \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 100 \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



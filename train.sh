DATA_PATH=XXXX
LANG=deen

MODEL_PATH=XXXX
IS_PRETRAINED=true

EXPERIMENT_NAME=XXXX
PROJECT_DIR=binaryalign
SAVE_PATH=$PROJECT_DIR/models/XXX
LOG_DIR=$PROJECT_DIR/logs/XXX


python train.py \
    --train_path $DATA_PATH/train/$LANG \
    --eval_path $DATA_PATH/dev/$LANG \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --log_dir $LOG_DIR \
    --name $EXPERIMENT_NAME \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --bs 8 \
    --warmup_ratio 0. \
    --logging_steps 100 \
    --threshold .5 \
    --do_train \
    --do_eval \
    --save_strategy end \
    --tk2word_prob max \
    --bidirectional_combine_type avg \
    --ignore_possible_alignments \
    --is_pretrained $IS_PRETRAINED
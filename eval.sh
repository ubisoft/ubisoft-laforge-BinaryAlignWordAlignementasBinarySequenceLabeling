DATA_PATH=XXXX
LANG=deen
MODEL_PATH=XXXX

python train.py \
    --eval_path $DATA_PATH/dev/$LANG \
    --model_path $MODEL_PATH \
    --threshold .5 \
    --do_eval \
    --tk2word_prob max \
    --bidirectional_combine_type avg \
    --is_pretrained 
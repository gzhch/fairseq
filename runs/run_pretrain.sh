TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=32          # Increase the batch size 16x

DATA_DIR=/blob/gzhch/data/wikitext-103

OUTPUT_PATH=/blob/gzhch/logs/pretrain/roberta-base/origin
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --save-dir $OUTPUT_PATH \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 | tee $OUTPUT_PATH/train_log.txt;

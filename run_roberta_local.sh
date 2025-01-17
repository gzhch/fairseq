TASK=$1
LR=$2
RFT=$3
LORA=$4
TYPE=2
SEED=$5
GRADED=linear

N_EPOCH=30
WARMUP_RATIO=15
BSZ=16        # Batch size.
MODEL=large

ROBERTA_PATH=../transformer/models/roberta.$MODEL/model.pt
DATA_PATH=../FastBERT/examples/roberta/glue/$TASK-bin/
#DATA_PATH=./$TASK-bin/
# ROBERTA_PATH=/blob/gzhch/model/roberta.$MODEL/model.pt
# DATA_PATH=/blob/gzhch/data/glue/$TASK-bin/

METRIC=accuracy
N_CLASSES=2
task_type=SMALL

if [ "$TASK" = "MNLI" ]
then
N_CLASSES=3
EPOCH_ITER=12471
task_type=LARGE
WARMUP_RATIO=60
fi

if [ "$TASK" = "QNLI" ]
then
EPOCH_ITER=3312
task_type=LARGE
WARMUP_RATIO=60
fi

if [ "$TASK" = "QQP" ]
then
EPOCH_ITER=11391
task_type=LARGE
WARMUP_RATIO=15
fi

if [ "$TASK" = "SST-2" ]
then
EPOCH_ITER=2105
task_type=LARGE
WARMUP_RATIO=60
fi

if [ "$TASK" = "MRPC" ]
then
EPOCH_ITER=115
fi

if [ "$TASK" = "RTE" ]
then
EPOCH_ITER=100
fi

if [ "$TASK" = "CoLA" ]
then
METRIC=mcc
EPOCH_ITER=268
fi

if [ "$TASK" = "STS-B" ]
then
METRIC=pearson_spearman
N_CLASSES=1
OPTION="--regression-target"
EPOCH_ITER=180
fi

if [ "$TASK" = "MNLI-10k" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
EPOCH_ITER=312
fi

if [ "$TASK" = "MNLI-5k" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
EPOCH_ITER=156
fi

if [ "$TASK" = "MNLI-3k" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
EPOCH_ITER=100
fi

EPOCH_ITER=$((EPOCH_ITER*2)) # expand to itr for bsz=16
BSZ_EXPAND=$((BSZ/16))
EPOCH_ITER=$((EPOCH_ITER/BSZ_EXPAND))
TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS/WARMUP_RATIO))
VALIDATE_INTERVAL=$((EPOCH_ITER/2))



#OUTPUT_PATH=/blob/gzhch/logs/${MODEL}/${TASK}/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
OUTPUT_PATH=tmp/out
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
# if [ -e $OUTPUT_PATH/train_log.txt ]; then
#     if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
#         echo "Training log existed"
#         exit 0
#     fi
# fi


python train.py $DATA_PATH \
    --graded-rft $GRADED \
    --lora $LORA \
    --random-ft $RFT \
    --mask-type $TYPE \
    --freeze-emb --freeze-norm \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $BSZ \
    --max-tokens 2200 \
    --seed $SEED \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_$MODEL \
    --criterion sentence_prediction \
    --num-classes $N_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_STEPS --warmup-updates $WARMUP_STEPS \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch $N_EPOCH \
    --find-unused-parameters \
    --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric \
    --no-save \
    --patience 8 \
    --num-workers 0 \
    --save-dir $OUTPUT_PATH \
    --no-progress-bar \
    --log-interval 100 \
    --no-epoch-checkpoints --no-last-checkpoints | tee $OUTPUT_PATH/train_log.txt; #    --train_bias;

#    --regression-target \

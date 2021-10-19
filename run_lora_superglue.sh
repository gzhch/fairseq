TASK=$1
LR=$2
RFT=$3
LORA=$4
TYPE=2
SEED=$5

#LR=5e-4             # Peak LR for polynomial LR scheduler.
N_EPOCH=50
WARMUP_RATIO=10
BSZ=32
VALID=val
MODEL=large

ROBERTA_PATH=../transformer/models/roberta.$MODEL/model.pt
#DATA_PATH=../FastBERT/examples/roberta/glue/$TASK-bin/
DATA_PATH=../transformer/datasets/superglue/$TASK
FAIRSEQ_USER_DIR=./superglue

TASK_NAME=sentence_prediction
CRITERION=sentence_prediction

METRIC=accuracy
N_CLASSES=2
task_type=SMALL

if [ "$TASK" = "MNLI" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
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

if [ "$TASK" = "WSC" ]
then
N_CLASSES=1
OPTION="--regression-target"
EPOCH_ITER=180
fi

if [ "$TASK" = "MultiRC" ]
then
EPOCH_ITER=1702
fi

if [ "$TASK" = "WiC" ]
then
EPOCH_ITER=375
TASK_NAME=wic
CRITERION=wic
fi

if [ "$TASK" = "CB" ]
then
ROBERTA_PATH=../transformer/models/roberta.$MODEL.mnli/model.pt
N_CLASSES=3
N_EPOCH=50
EPOCH_ITER=16
fi

if [ "$TASK" = "BoolQ" ]
then 
EPOCH_ITER=590
VALID=valid
fi

# EPOCH_ITER=$((EPOCH_ITER*2)) # expand to itr for bsz=16
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
    --valid-subset $VALID \
    --lora $LORA \
    --random-ft $RFT \
    --mask-type $TYPE \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $BSZ \
    --max-tokens 2200 \
    --seed $SEED \
    --bpe gpt2 \
    --task $TASK_NAME --criterion $CRITERION \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_$MODEL \
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
    --save-dir $OUTPUT_PATH \
    --user-dir $FAIRSEQ_USER_DIR \
    --no-progress-bar \
    --log-interval 100 \
    --num-workers 0 \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints --no-last-checkpoints | tee $OUTPUT_PATH/train_log.txt; #    --train_bias;

#    --regression-target \
#    --patience 8 \
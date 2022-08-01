pip install --editable ./ -t ./package



TASK=MRPC
SUBSET=mnliV2
LR=1e-4
SEED=2

SUBNET_PATH_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gzhch/Data/subnet
#PREFIX=subnetV3
PREFIX=tmp
#PREFIX=tmp_fft

N_EPOCH=30
WARMUP_RATIO=15
BSZ=16        # Batch size.
UPDATE_FREQ=1
MODEL=large

SUBNET_PATH=$SUBNET_PATH_ROOT/$SUBSET.pkl

ROBERTA_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gzhch/Data/Models/roberta.$MODEL/model.pt
DATA_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gzhch/Data/Datasets/glue/$TASK-bin/

METRIC=accuracy
N_CLASSES=2
task_type=SMALL

if [ "$TASK" = "MNLI" ]
then
N_CLASSES=3
EPOCH_ITER=12471
N_EPOCH=4

task_type=LARGE
WARMUP_RATIO=60
fi

if [ "$TASK" = "QNLI" ]
then
N_EPOCH=10

EPOCH_ITER=3312
task_type=LARGE
WARMUP_RATIO=60
fi

if [ "$TASK" = "QQP" ]
then
N_EPOCH=10

EPOCH_ITER=11391
task_type=LARGE
WARMUP_RATIO=15
fi

if [ "$TASK" = "SST-2" ]
then
N_EPOCH=10
EPOCH_ITER=2105
task_type=LARGE
WARMUP_RATIO=30
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
EPOCH_ITER=$((EPOCH_ITER*16/(BSZ*UPDATE_FREQ)))
TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS/WARMUP_RATIO))
VALIDATE_INTERVAL=$((EPOCH_ITER/2))



#OUTPUT_PATH=/blob/gzhch/logs/${MODEL}/${TASK}/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
OUTPUT_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gzhch/Log/mp/${MODEL}/${TASK}/$PREFIX-$SUBSET-$LR-$SEED
#OUTPUT_PATH=tmp/out
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
# if [ -e $OUTPUT_PATH/train_log.txt ]; then
#     if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
#         echo "Training log existed"
#         exit 0
#     fi
# fi


python3 train.py $DATA_PATH \
    --subnet-path $SUBNET_PATH \
    --lora 0 \
    --random-ft 0 \
    --freeze-emb \
    --freeze-norm \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $BSZ \
    --batch-size $BSZ \
    --update-freq $UPDATE_FREQ \
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
    --max-epoch $N_EPOCH \
    --find-unused-parameters \
    --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric \
    --patience 8 \
    --num-workers 0 \
    --save-dir $OUTPUT_PATH \
    --no-progress-bar \
    --no-save \
    --log-interval 100 \
    --no-epoch-checkpoints --no-last-checkpoints | tee $OUTPUT_PATH/train_log.txt; #    --train_bias;

#    --regression-target \
#    --no-save \
#    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \


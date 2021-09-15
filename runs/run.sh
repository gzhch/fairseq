TASK=$1
LR=$2
TYPE=$3


#bash ./run_roberta_large.sh $TASK $LR 1e-3 $TYPE 1
#bash ./run_roberta_large.sh $TASK $LR 5e-3 $TYPE 1
#bash ./run_roberta_large.sh $TASK $LR 1e-2 $TYPE 1
#bash ./run_roberta_large.sh $TASK $LR 5e-2 $TYPE 1

#bash ./run_roberta_large.sh $TASK $LR 1e-3 $TYPE 2
#bash ./run_roberta_large.sh $TASK $LR 5e-3 $TYPE 2
#bash ./run_roberta_large.sh $TASK $LR 1e-2 $TYPE 2
#bash ./run_roberta_large.sh $TASK $LR 5e-2 $TYPE 2

bash ./run_roberta_large.sh $TASK $LR 1e-3 $TYPE 3
bash ./run_roberta_large.sh $TASK $LR 5e-3 $TYPE 3
bash ./run_roberta_large.sh $TASK $LR 1e-2 $TYPE 3
bash ./run_roberta_large.sh $TASK $LR 5e-2 $TYPE 3

bash ./run_roberta_large.sh $TASK $LR 1e-3 $TYPE 4
bash ./run_roberta_large.sh $TASK $LR 5e-3 $TYPE 4
bash ./run_roberta_large.sh $TASK $LR 1e-2 $TYPE 4
bash ./run_roberta_large.sh $TASK $LR 5e-2 $TYPE 4

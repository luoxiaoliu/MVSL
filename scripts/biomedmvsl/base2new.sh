#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
# custom config

start_seed=1
end_seed=3
TASK_NAME=$1

DATA=/home/zrway/willow/BiomedMVSL/data
#DATASETS=(btmri chmnist)
DATASETS=(btmri busi chmnist covid ctkidney dermamnist kneexray kvasir lungcolon octmnist retina)
MODEL=BiomedCLIP
METHOD=BiomedMVSL
TRAINER=BiomedMVSL_${MODEL}

SHOTS=16
LOADEP=50
SUB_base=base
SUB_novel=new

BASE_PATH=$2/base2new
for DATASET in "${DATASETS[@]}"; do
    COMMON_DIR=${DATASET}
    TRAIN_DIR=$BASE_PATH/train_${SUB_base}/$COMMON_DIR
    TEST_DIR=$BASE_PATH/test_${SUB_novel}/$COMMON_DIR
    for ((SEED=${start_seed}; SEED<=${end_seed}; SEED++)); do
        DIR=$TRAIN_DIR/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB_base}
        fi
        MODEL_DIR=$TRAIN_DIR/seed${SEED}
        DIR=$TEST_DIR/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
        fi
    done
    python parse_test_res.py --directory $TRAIN_DIR --test-log 2>&1 | tee $TRAIN_DIR/summary.txt
    python parse_test_res.py --directory $TEST_DIR --test-log 2>&1 | tee $TEST_DIR/summary.txt
done
python parse_test_res.py --directory $BASE_PATH/train_${SUB_base} --test-log --multi-exp 2>&1 | tee $BASE_PATH/train_${SUB_base}/summary.txt
python parse_test_res.py --directory $BASE_PATH/test_${SUB_novel} --test-log --multi-exp 2>&1 | tee $BASE_PATH/test_${SUB_novel}/summary.txt
python parse_test_res.py --directory $BASE_PATH --hm 2>&1 | tee $BASE_PATH/summary.txt

# cleaning
# find "$BASE_PATH/train_${SUB_base}" -mindepth 1 ! -name 'summary.txt' -exec rm -rf {} +
# find "$BASE_PATH/test_${SUB_novel}" -mindepth 1 ! -name 'summary.txt' -exec rm -rf {} +


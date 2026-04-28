#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

start_seed=1
end_seed=3

TASK_NAME=$1

# custom config
DATA=/home/zrway/willow/BiomedMVSL/data
#SHOTS=(1 2)
#DATASETS=(btmri busi)
SHOTS=(1 2 4 8 16)
DATASETS=(btmri busi chmnist covid ctkidney dermamnist kneexray kvasir lungcolon octmnist retina)

MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end

METHOD=BiomedMVSL
TRAINER=BiomedMVSL_${MODEL}

ROOT_PATH=$2/few_shot
for SHOT in ${SHOTS[@]}; do
    BASE_PATH=${ROOT_PATH}/shots_${SHOT}
    for DATASET in "${DATASETS[@]}"; do
        BASE_DIR=$BASE_PATH/${DATASET}
        for ((SEED=${start_seed}; SEED<=${end_seed}; SEED++)); do
            DIR=$BASE_DIR/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
            else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
                --output-dir ${DIR} \
                TRAINER.BIOMEDMVSL.N_CTX ${NCTX} \
                TRAINER.BIOMEDMVSL.CSC ${CSC} \
                TRAINER.BIOMEDMVSL.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOT}
            fi
        done
        # prints averaged results
        python parse_test_res.py --directory $BASE_DIR --test-log 2>&1 | tee $BASE_DIR/summary.txt

        # cleaning weights
        # rm -rf $BASE_DIR/seed*
    done
    python parse_test_res.py --directory $BASE_PATH --multi-ds 2>&1 | tee $BASE_PATH/summary.txt
done
python parse_test_res.py --directory $ROOT_PATH --all-res 2>&1 | tee $ROOT_PATH/summary.txt


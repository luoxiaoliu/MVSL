status=${1:-run}

if [ "$status" == "stop" ]; then
  pkill -f "base2new.sh"
  pkill -f "few_shot.sh"
  pkill -f "batch_all.sh"
  exit 0
fi

task_name="output_moe_5_0_15_0_18_5"
out_path=output/${task_name}

mkdir -p ${out_path}

bash ./scripts/biomedcoop/base2new.sh ${task_name} ${out_path} > ${out_path}/out_base2new.log 2>&1 &

bash ./scripts/biomedcoop/few_shot.sh ${task_name} ${out_path} > ${out_path}/out_few_shot.log 2>&1 &
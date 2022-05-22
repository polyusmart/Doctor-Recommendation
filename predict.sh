gpu_id=0
n_gpu=8
for seed in 131 725 1104
do
    for name in mul_att_full mul_att_wo_d mul_att_wo_p mul_att_wo_sl
    do
        echo "randome seed: ${seed}, experiment name: ${name}, run on gpu ${gpu_id}"
        output_dir="output/seed_${seed}/${name}"

        # change exp settings here, 
        # corresponding arguments are in utils/config.py
        if [[ "${name}" == "mul_att_full" ]]
        then 
            embeddings_path=bert_embeddings
            add_self_att_on="none"
        elif [[ "${name}" == "mul_att_wo_d" ]]
        then
            embeddings_path=bert_embeddings
            add_self_att_on="profile"
        elif [[ "${name}" == "mul_att_wo_p" ]]
        then
            embeddings_path=bert_embeddings
            add_self_att_on="dialogs"
        elif [[ "${name}" == "mul_att_wo_sl" ]]
        then
            embeddings_path=bert_embeddings_wo_sl
            add_self_att_on="none"
        fi

        echo "${name} prediction start!"
        nohup python -u predict.py \
            -seed $seed \
            -gpu $gpu_id \
            -name $name \
            -embeddings_path $embeddings_path \
            -add_self_att_on $add_self_att_on \
            -output_dir $output_dir > ${name}_seed${seed}_prediction.log 2>&1 &

        gpu_id=$(( ($gpu_id + 1) % $n_gpu))
    done
done
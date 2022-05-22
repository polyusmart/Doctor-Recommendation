for name in mul_att_full mul_att_wo_d mul_att_wo_p mul_att_wo_sl
do
    for seed in 131 725 1104
    do
        output_dir="output/seed_${seed}/${name}"
        # use ranklib to evaluate prediction results
        for epoch in best 
        do
            echo "==>Evaluate ${output_dir} epoch ${epoch}<=="
            python ./sort_by_score.py -res_dir $output_dir -epoch ${epoch}_model.pt
            for metric in p@1 map err@5
            do
                java -jar ./RankLib-2.16.jar -test "${output_dir}/sorted_test_${epoch}_model.pt.dat" \
                    -metric2T $metric -idv "${output_dir}/${metric}_${epoch}.txt"
            done
        done
    done
done
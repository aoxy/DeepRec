swapon -s
swapoff -a
swapon /home/axy/code/swapfile
swapon -s

# bash start_exps.sh "DLRM" "directio"
# mv archives archives_DLRM_directio

bash start_exps.sh "MMoE" "directio"
mv archives archives_MMoE_directio

bash start_exps.sh "WDL" "directio"
mv archives archives_WDL_directio

bash start_exps.sh "DIEN" "directio"
mv archives archives_DIEN_directio

# sudo nohup ./multi_models.sh > script_output_multi_models.log 2>&1 &
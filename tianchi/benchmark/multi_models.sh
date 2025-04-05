
docker stop axynetp
docker update --memory "160g" --memory-swap "170g" axynetp
docker start axynetp
# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/reinstall_deeprc.sh"

swapon -s
swapoff -a
swapon /home/axy/code/swapfile
swapon -s

bash start_exps.sh "DLRM" "directio"
mv archives archives_DLRM_directio

bash start_exps.sh "MMoE" "directio"
mv archives archives_MMoE_directio

bash start_exps.sh "WDL" "directio"
mv archives archives_WDL_directio

# bash start_exps.sh "DIEN" "directio"
# mv archives archives_DIEN_directio

# sudo nohup ./multi_models.sh > script_output_multi_3m.log 2>&1 &

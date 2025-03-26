#!/bin/bash
# nohup ./start.sh > script_output_4m.log 2>&1 &

swapon -s
swapoff -a
swapon /home/axy/code/swapfile
swapon -s


docker stop axynetp
docker start axynetp
docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/reinstall_deeprc.sh"

docker stop axynetp
docker start axynetp
docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/tianchi/run_models.sh"

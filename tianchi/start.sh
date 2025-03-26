#!/bin/bash
# nohup ./start.sh > script_output_4m.log 2>&1 &


docker stop axynetp
docker update --memory "560g" --memory-swap "570g" axynetp
docker start axynetp
docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/reinstall_deeprc.sh"

docker stop axynetp
docker start axynetp
docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/tianchi/run_models.sh"

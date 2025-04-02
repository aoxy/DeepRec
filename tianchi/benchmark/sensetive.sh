# cp /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py.bakv2

swapon -s
swapoff -a
swapon /home/axy/code/swapfile
swapon -s

sed  's/TF_SSDHASH_IO_SCHEME_VALUE/directio/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py.bakv2 > /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
sed -i 's/AAAAAAAAAAAAAAAB32LFUBBBBBBBBBBB/B32LFU/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
bash start_exps.sh "DLRM" "DLRM_directio"
mv archives archives_directio

sed  's/TF_SSDHASH_IO_SCHEME_VALUE/directio/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py.bakv2 > /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
sed -i 's/AAAAAAAAAAAAAAAB32LFUBBBBBBBBBBB/LFU/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
bash start_exps.sh "DLRM" "DLRM_lfu"
mv archives archives_lfu

sed  's/TF_SSDHASH_IO_SCHEME_VALUE/mmap_and_madvise/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py.bakv2 > /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
sed -i 's/AAAAAAAAAAAAAAAB32LFUBBBBBBBBBBB/B32LFU/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
bash start_exps.sh "DLRM" "DLRM_mmap_and_madvise"
mv archives archives_mmap_and_madvise

sed  's/TF_SSDHASH_IO_SCHEME_VALUE/mmap/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py.bakv2 > /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
sed -i 's/AAAAAAAAAAAAAAAB32LFUBBBBBBBBBBB/B32LFU/g' /home/axy/code/aoxy/DeepRec/tianchi/DLRM/train.py
bash start_exps.sh "DLRM" "DLRM_mmap"
mv archives archives_mmap

# sudo nohup ./sensetive.sh > script_output_sensetive.log 2>&1 &

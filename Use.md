## 进入容器

```shell
sudo docker stop axynetp
sudo docker start axynetp
sudo docker exec -it axynetp /bin/bash
cd /home/code/aoxy/DeepRec/
cd /home/code/aoxy/DeepRec/tianchi/DLRM/
bash test.sh &
```

## 限制容器资源

```shell
sudo docker update --memory "16g" --memory-swap "29g" axynetp
sudo docker update --memory "13g" --memory-swap "16g" axynetp
sudo docker update --memory "8g" --memory-swap "21g" axynetp
sudo docker update --memory "160g" --memory-swap "280g" axynetp
# 查看限制
sudo docker inspect axynetp --format='{{.HostConfig.Memory}}'
sudo docker inspect axynetp --format='{{.HostConfig.MemorySwap}}'
```

## 设置Swap

```shell
sudo swapon -s
sudo swapoff -a
sudo dd if=/dev/zero of=/home/axy/code/swapfile bs=1024 count=$((13 * 1024 * 1024))
sudo chmod 600 /home/axy/code/swapfile
sudo mkswap /home/axy/code/swapfile
sudo swapon /home/axy/code/swapfile
```

## 模型测试

```shell
rm -rf /home/code/aoxy/DeepRec/tianchi/DLRM/result
rm /tmp/ssd_utpy/*
rep=0702v1
mkdir -p metrics_$rep/dlrm_log_files/
cache_cap=5
python3 train_s.py --data_location=/home/code/elem --smartstaged=True --cache_cap=$cache_cap 1> metrics_$rep/dlrm_log_files/dlrm_log_$cache_cap.txt 2>&1
python3 train.py --data_location=/home/code/elem --smartstaged=True --cache_cap=5 1> dlrm_log.txt 2>&1
python3 train.py --data_location=/home/code/elem --smartstaged=True --no_eval
```

## 性能测试

```shell
perf record -F 999 ./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingvariableTest.TestBLFUCachePrefetch
perf report
gdb ./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test core-embedding_varia
```

## 编译

```shell
bazel build -c opt --config=opt //tensorflow/python:embedding_variable_ops_test
./bazel-bin/tensorflow/python/embedding_variable_ops_test

bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_performance_test
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test

bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_ops_test
./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test

bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_memory_test
./bazel-bin/tensorflow/core/kernels/embedding_variable_memory_test

./bazel-bin/tensorflow/python/embedding_variable_ops_test EmbeddingVariableTest.testEmbeddingVariableForDRAMAndSSD testEmbeddingVariableForMultiTierInference
./bazel-bin/tensorflow/python/embedding_variable_ops_test 2>py2.txt >py1.txt
./bazel-bin/tensorflow/python/embedding_variable_ops_test EmbeddingVariableTest.testEmbeddingVariableForDRAMAndLEVELDB > pylog.txt 2>&1
./bazel-bin/tensorflow/python/embedding_variable_ops_test EmbeddingVariableTest.testEmbeddingVariableForDRAMAndSSDSaveCkpt > pylog.txt 2>&1

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestBLFUCachePrefetch
./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=KVInterfaceTest.TestSSDKVSyncCompaction

./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingvariableTest.TestBLFUCachePrefetch

./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestLookupOrCreateElastic
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable

heaptrack ./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEviction
```

## PY代码修改

```shell
cp /home/code/aoxy/DeepRec/tensorflow/python/ops/kv_variable_ops.py /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/ops/kv_variable_ops.py
cp /home/code/aoxy/DeepRec/tensorflow/python/training/adam_async.py /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/training/adam_async.py
```

## 配置新物理机

```shell
ssh-keygen -b 4096 -t rsa
# ssh-copy-id root@IP

# CentOS
yum install htop
dnf install git-all
dnf remove docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-selinux docker-engine-selinux docker-engine
dnf -y install dnf-plugins-core
yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
yum install docker-ce docker-ce-cli containerd.io

# Ubuntu
apt update
apt-get install htop -y
apt-get install git-all -y
apt-get remove docker docker-engine docker.io containerd runc
apt-get install ca-certificates curl gnupg lsb-release -y
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
apt-get install docker-ce docker-ce-cli containerd.io -y

systemctl start docker
mkdir -p /home/aoxuyang/code/aoxy
mkdir -p /home/aoxuyang/code/elem/
cd /home/aoxuyang/code/aoxy
cat ~/.ssh/id_rsa.pub
git clone git@github.com:aoxy/DeepRec.git
scp ./Use.md root@IP:/home/aoxuyang/code/aoxy/DeepRec/
scp ./get_url.py root@IP:/home/aoxuyang/code/elem/
scp ./generate.sh root@IP:/home/aoxuyang/code/elem/
# [***Ali ELM dataset***](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131047)
docker run -it --name axynetp --privileged --net=host -v /home/aoxuyang/code:/home/code alideeprec/deeprec-build:deeprec-dev-cpu-py38-ubuntu20.04
```

```shell
ulimit -c unlimited
echo '1' > /proc/sys/kernel/core_uses_pid
echo "./core-%e-%p-%t"> /proc/sys/kernel/core_pattern
apt-get update
apt-get install gdb -y
```

```shell
git config --global user.name "aoxy"
git config --global user.email "jerryao@mail.ustc.edu.cn"
```

## `reinstall_deeprc.sh`

```shell
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
if [ $? -eq 0 ]; then
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2306-cp38-cp38-linux_x86_64.whl --force-reinstall
    pip uninstall protobuf -y
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.19.0
fi
```

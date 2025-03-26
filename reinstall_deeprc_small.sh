cd /home/code/aoxy/DeepRec/
export BAZEL_JAVAC_OPTS="-J-Xmx12g"

bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=12288 --local_cpu_resources=4

if [ $? -eq 0 ]; then
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2306-cp38-cp38-linux_x86_64.whl --force-reinstall
    pip uninstall protobuf -y
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.19.0
fi


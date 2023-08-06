docker start axynet

docker exec -it axynet /bin/bash

cd /home/code/aoxy/DeepRec/

bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_ops_test

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestLookupConcurrencyCache

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestLookupConcurrencyCacheTaoBao

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestCacheTaoBaoBatch  2>result/3/cpplog2.txt >result/3/cpplog1.txt

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestCacheTaoBaoNoEvicBatch 2>result/4/cpplog2.txt >result/4/cpplog1.txt

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestCacheTaoBaoBatch  2>result2/5/cpplog2.txt >result2/5/cpplog1.txt

./bazel-bin/tensorflow/core/kernels/embedding_variable_ops_test --gtest_filter=EmbeddingVariableTest.TestCacheTaoBaoNoEvicBatch 2>result2/8/cpplog2.txt >result2/8/cpplog1.txt
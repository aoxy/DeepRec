cd /home/code/aoxy/DeepRec/
mkdir -p ./tianchi/benchmark/cache_test_data
bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_performance_test
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./tianchi/benchmark/cache_test_data/data1.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./tianchi/benchmark/cache_test_data/data2.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./tianchi/benchmark/cache_test_data/data3.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./tianchi/benchmark/cache_test_data/data4.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./tianchi/benchmark/cache_test_data/data5.txt >py1.txt
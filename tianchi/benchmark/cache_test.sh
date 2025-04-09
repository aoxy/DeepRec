mkdir ./cache_test_data/
bazel build -c opt --config=opt //tensorflow/core/kernels:embedding_variable_performance_test
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./cache_test_data/data1_41hohit.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./cache_test_data/data2_41hohit.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./cache_test_data/data3_41hohit.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./cache_test_data/data4_41hohit.txt >py1.txt
./bazel-bin/tensorflow/core/kernels/embedding_variable_performance_test --gtest_filter=EmbeddingVariablePerformanceTest.TestCacheUpdateAndEvictionTable  2>./cache_test_data/data5_41hohit.txt >py1.txt
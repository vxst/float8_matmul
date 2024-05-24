NVCC = nvcc

CFLAGS = -DTEST

TEST_ADD_CU = test_add.cu
TEST_FMA_CU = test_fma.cu
TEST_MATMUL_CU = test_matmul.cu
KERNEL_CU = ../matmulf8_kernel.cu ../matmulf8.cu ../load_core.cu
OUTPUT_ADD = test_add
OUTPUT_FMA = test_fma
OUTPUT_MATMUL = test_matmul
BIN_FILE = test_add.bin test_fma.bin

OUTPUT_EXE = $(OUTPUT_ADD) $(OUTPUT_FMA) $(OUTPUT_MATMUL)

test: $(OUTPUT_EXE) $(BIN_FILE)
	./$(OUTPUT_ADD)
	./$(OUTPUT_FMA)
	./$(OUTPUT_MATMUL)

$(OUTPUT_ADD): $(TEST_ADD_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_ADD_CU) $(KERNEL_CU) -o $(OUTPUT_ADD)

$(OUTPUT_FMA): $(TEST_FMA_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_FMA_CU) $(KERNEL_CU) -o $(OUTPUT_FMA)

$(OUTPUT_MATMUL): $(TEST_MATMUL_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_MATMUL_CU) $(KERNEL_CU) -o $(OUTPUT_MATMUL)

$(BIN_FILE): gen_test_bin.py
	python gen_test_bin.py

clean:
	rm -f $(OUTPUT_EXE) $(BIN_FILE)

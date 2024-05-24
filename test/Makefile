NVCC = nvcc

CFLAGS = -DTEST

TEST_ADD_CU = test_add.cu
TEST_FMA_CU = test_fma.cu
KERNEL_CU = ../matmulf8_kernel.cu
OUTPUT_ADD = test_add
OUTPUT_FMA = test_fma
BIN_FILE = test_add.bin test_fma.bin

OUTPUT_EXE = $(OUTPUT_ADD) $(OUTPUT_FMA)

all: $(OUTPUT_EXE) $(BIN_FILE)

$(OUTPUT_ADD): $(TEST_ADD_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_ADD_CU) $(KERNEL_CU) -o $(OUTPUT_ADD)

$(OUTPUT_FMA): $(TEST_FMA_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_FMA_CU) $(KERNEL_CU) -o $(OUTPUT_FMA)

$(BIN_FILE): gen_test_bin.py
	python gen_test_bin.py

clean:
	rm -f $(OUTPUT_EXE) $(BIN_FILE)

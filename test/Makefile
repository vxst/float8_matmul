NVCC = nvcc

CFLAGS = -DTEST

TEST_ADD_CU = test_add.cu
KERNEL_CU = ../matmulf8_kernel.cu
OUTPUT_EXE = test_add
BIN_FILE = test_add.bin

all: $(OUTPUT_EXE) $(BIN_FILE)

$(OUTPUT_EXE): $(TEST_ADD_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(TEST_ADD_CU) $(KERNEL_CU) -o $(OUTPUT_EXE)

$(BIN_FILE): gen_test_bin.py
	python gen_test_bin.py

clean:
	rm -f $(OUTPUT_EXE) $(BIN_FILE)

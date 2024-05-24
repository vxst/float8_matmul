NVCC = nvcc

CFLAGS = -O3 -arch=native

MAIN_CU = main.cu
KERNEL_CU = load_core.cu matmulf8.cu matmulf8_kernel.cu
OUTPUT_BENCH = bench

$(OUTPUT_BENCH): $(MAIN_CU) $(KERNEL_CU)
	$(NVCC) $(CFLAGS) $(MAIN_CU) $(KERNEL_CU) -o $(OUTPUT_BENCH)

test: $(OUTPUT_EXE)
	cd test && $(MAKE) test

clean:
	rm -f $(OUTPUT_BENCH)
	cd test && $(MAKE) clean

.PHONY: clean test
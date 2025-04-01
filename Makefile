COMPILER = nvcc

%: %.cu
	$(COMPILER) -o $@ $<
	./$@

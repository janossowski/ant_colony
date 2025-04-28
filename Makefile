# Absolute path to CUDA installation
CUDA_PATH = /usr/local/cuda

# Compiler and flags
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -arch=sm_60 -O2
INCLUDES = -I$(CUDA_PATH)/include
LIBRARIES = -L$(CUDA_PATH)/lib64 -lcudart -lcurand

# Files
SRC = main.cu
OBJ = $(SRC:.cu=.o)
EXE = my_program

# Default target
all: $(EXE)

# How to build the executable
$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBRARIES)

# How to compile .cu into .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ) $(EXE)

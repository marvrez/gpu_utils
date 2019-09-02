DEBUG  ?= 0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61] \
      -gencode arch=compute_70,code=[sm_70,compute_70] \
      -gencode arch=compute_75,code=[sm_75,compute_75]

OBJ= main.o device_query.o
EXECOBJA= 

VPATH=./src/
EXEC=gpu_utils
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

# include CUDA
COMMON+= -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

EXECOBJS = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS   = $(addprefix $(OBJDIR), $(OBJ))
DEPS   = $(wildcard include/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
		$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean
clean:
	rm -rf $(OBJS) $(ALIB) $(EXEC) $(EXECOBJS) $(OBJDIR)/* $(OBJDIR)

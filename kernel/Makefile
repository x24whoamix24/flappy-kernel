# SPDX-License-Identifier: Apache-2.0
# Builds the CUDA tools that run the co-tenant workloads without the LLM.
#   standalone  game + trainer alone; gradient check; HBM probe; policy export;
#               shared-memory layout for server/ (`standalone layout`)
#   separate    trainer and game as separate processes (comparison baseline)
NVCC ?= nvcc
ARCH ?= sm_90a
FLAGS ?= -O3 -std=c++17 -lineinfo
HEADERS = cotenant.cuh $(wildcard core/* training/* game/* host/*) tools/launch.cuh

all: standalone separate

standalone: tools/standalone.cu $(HEADERS)
	$(NVCC) $(FLAGS) -arch=$(ARCH) $< -o $@

separate: tools/separate.cu $(HEADERS)
	$(NVCC) $(FLAGS) -arch=$(ARCH) $< -o $@

clean:
	rm -f standalone separate

.PHONY: all clean

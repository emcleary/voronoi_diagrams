#!/bin/env bash

# difficult to run over larger inputs; likely just precision issues
# python3 -O scripts/run_circle_of_sites.py 1\
# 	4 8 16 32 64 128 256 512 1024

python3 -O scripts/run_random_sites.py 20\
	4 8 16 32 64 128 256 512 1024 2048 4096\
	8192 16384 32768 49152 65536 81920 98304\
	114688 131072


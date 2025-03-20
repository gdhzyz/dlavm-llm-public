VLOG = /tools/Xilinx/Vivado/2023.2/bin/xvlog
VSIM = /tools/Xilinx/Vivado/2023.2/bin/xsim
VELAB = /tools/Xilinx/Vivado/2023.2/bin/xelab
TOP_MODULE = testbench_HBM_MVM
SIM_DEFINE = 
LIB_PATH = work

all: run

run :
	${VLOG} $(TOP_MODULE).sv $(SIM_DEFINE) --sv -i .
	${VELAB} -debug all --snapshot work ${TOP_MODULE} --timescale 1ns/1ps
	${VSIM} --runall work

.PHONY: run all

VLIB = vlib.exe
VLOG = vlog.exe
VSIM = vsim.exe
TOP_MODULE = testbench_HBM_MVM
SIM_DEFINE = 
LIB_PATH = work

all: run

run : ${LIB_PATH}
	${VLOG} $(TOP_MODULE).sv $(SIM_DEFINE) +incdir+.
	${VSIM} -c -L work ${TOP_MODULE} -do "run -all; quit"	

${LIB_PATH} :
	${VLIB} ${LIB_PATH}


.PHONY: run all
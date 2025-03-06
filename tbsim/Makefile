TESTBENCH := testbench
NAME      := $(shell date +"%Y_%m%d")
PREFIX    := workspace
WORK_DIR  := ${PREFIX}_${NAME}
TEMP_DIR  := temp
PWD_DIR   := $(shell pwd)


all:RUN


RUN:WORK_DIR ${TEMP_DIR}
	@${PWD_DIR}/bin/install ${TESTBENCH} ${WORK_DIR} ${TEMP_DIR}
	@${PWD_DIR}/bin/process ${TEMP_DIR} ${WORK_DIR}
	@rm -r ${TEMP_DIR}

WORK_DIR:
	@mkdir -p ${WORK_DIR}
	@ln -s ${PWD_DIR}/vivado.Makefile ${WORK_DIR}/Makefile

${TEMP_DIR}:
	@mkdir -p ${TEMP_DIR}

clean:
	rm -rf ${WORK_DIR}
	rm -rf ${TEMP_DIR}

.PHONY: RUN NAME


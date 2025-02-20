SRCS := $(wildcard *.cc)
INCS := -I./
TAGS := mod.so

all: ${TAGS}

${TAGS}: ${SRCS}
	g++ -shared -fPIC $^ ${INCS} -o $@

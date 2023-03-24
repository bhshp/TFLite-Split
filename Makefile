src_dir = ./src
build_dir = ./build
include_dir = ./include

target = $(build_dir)/split_tflite

CC = g++
CCFLAGS = -g -I./${include_dir} -I./third-party -lfmt -std=c++23 -Wall -Wextra -Werror -pedantic-errors -O2

src = $(wildcard $(src_dir)/*.cc)
object = $(patsubst $(src_dir)/%.cc,$(build_dir)/%.o,$(src))

all:
	mkdir -p ${build_dir}
	make clean
	make ${target}

${target}: ${object}
	${CC} -o $@ $^ ${CCFLAGS}

$(build_dir)/%.o: $(src_dir)/%.cc
	$(CC) -c -o $@ $^ $(CCFLAGS)

clean:
	${RM} -rf ${build_dir}/*

.PHONY: all clean

CC=g++
CFLAGS=-I. -I/usr/include/freetype2 -I/usr/include/libpng16 -Wall \
	-L/usr/local/lib -lfreetype
PBFLAGS=-shared -std=c++11 -fPIC $(shell python3 -m pybind11 --includes)
DEPS = $(wildcard *.h)
SRCS = $(wildcard *.cpp)
# SRCS := $(filter-out renderer.cpp, $(SRCS))
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
target=renderer$(shell python3-config --extension-suffix)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(PBFLAGS)

all: $(target) # renderer

renderer: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

$(target): $(OBJS)
	$(CC) -o $(target) $^ $(CFLAGS) $(PBFLAGS)
	rm *.o

.PHONY : clean
clean:
	rm renderer renderer.cpython*
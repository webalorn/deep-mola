CC=g++
CFLAGS=-W -Wall -ansi -pedantic -O3 -std=c++14
LDFLAGS=
EXEC=deepMola
SRC= bin/main.cpp bin/layer_network.cpp bin/types.cpp
HPP= src/headers/layer_network.hpp src/headers/types.hpp
OBJ= $(SRC:.cpp=.o)

all: $(EXEC)

deepMola: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

bin/main.o: src/main.cpp
	@mkdir -p bin
	$(CC) -o $@ -c $< $(CFLAGS)

bin/%.o: src/cpp/%.cpp src/headers/%.hpp
	@mkdir -p bin
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf bin

mrproper: clean
	rm -rf $(EXEC)
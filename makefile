objects = main.o function.o
all: $(objects)
	nvcc $(objects) -o app
%.o: %.cpp
	nvcc -x cu -g -lineinfo -dc $< -o $@
clean:
	rm -f *.o app

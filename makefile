objects = main.o function.o
all: $(objects)
	nvcc -arch=sm_61 $(objects) -o app
%.o: %.cpp
	nvcc -x cu -arch=sm_61 -g -lineinfo -dc $< -o $@
clean:
	rm -f *.o app

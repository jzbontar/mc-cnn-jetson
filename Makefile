main: main.o stereo.o
	c++ main.o stereo.o -o main -L/usr/local/zed/lib  -L/usr/local/cuda-6.5/lib -lsl_zed /usr/local/cuda-6.5/lib/libcudart.so -lGL -lglut -lcudnn -Wl,-rpath,/usr/local/zed/lib:/usr/local/cuda-6.5/lib

main.o: main.cpp
	c++ -O2 -I/usr/local/zed/include -I/usr/local/cuda-6.5/include -std=c++0x -o main.o -c main.cpp

stereo.o: stereo.cu
	nvcc -arch sm_32 -O2 -o stereo.o -c stereo.cu

clean:
	rm -f main *.o *.so

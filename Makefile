main: main.o
	c++ main.o -o main -L/usr/local/zed/lib  -L/usr/local/cuda-6.5/lib -lsl_zed /usr/local/cuda-6.5/lib/libcudart.so -lGL -lglut -Wl,-rpath,/usr/local/zed/lib:/usr/local/cuda-6.5/lib

main.o: main.cpp
	c++ -O2 -I/usr/local/zed/include -I/usr/local/cuda-6.5/include -std=c++0x -o main.o -c main.cpp


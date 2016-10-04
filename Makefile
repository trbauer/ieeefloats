default: fp16.exe

fp16.exe: fp16.cpp
	g++ fp16.cpp -o fp16.exe -std=c++11

test: ./fp16.exe
	@./fp16.exe
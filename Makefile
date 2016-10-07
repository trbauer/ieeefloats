default: fp16.exe

fp16.exe: fp16.cpp
	g++ fp16.cpp -o fp16.exe -std=c++11

clean:
	@rm -f fp16.exe
	
test: ./fp16.exe
	@./fp16.exe
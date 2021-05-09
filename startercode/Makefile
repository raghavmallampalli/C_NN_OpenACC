all: driver.o convNet.o
	gcc driver.o convNet.o -pthread -Wall -o run

driver.o: driver.c
	gcc -c -Wall driver.c

convNet.o: convNet.c
	gcc -c -Wall convNet.c

clean:
	rm -rf *.o run
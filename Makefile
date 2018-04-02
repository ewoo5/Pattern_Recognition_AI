CC=g++
CFLAGS=-I -Wall -Werror -g -std=c++11

all: naive_bayes.o jarvis.o 
	$(CC) -o jarvis jarvis.o naive_bayes.o $(CFLAGS)

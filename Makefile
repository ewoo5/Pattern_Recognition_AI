CC=g++
CFLAGS=-I -Wall -Werror -g -std=c++11

all: perceptron.o naive_bayes.o jarvis.o 
	$(CC) -o jarvis jarvis.o naive_bayes.o perceptron.o $(CFLAGS)

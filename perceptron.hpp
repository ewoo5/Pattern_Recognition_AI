#include <iostream>
#include <fstream>
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

class perceptron{

	private:
		int width, height;
		float **** vec_w; // in the format vec[number class][row][column][0-Weight; 1-tokens].
		float eta; // Used to mutiply with apha/x
		std::ifstream training_txt, testing_txt;		

	public:
		perceptron();
		perceptron(int in_width, int in_height, char * in_train, char * in_test);
		~perceptron();
		
		void train(int epochs);
		void test();
		void randomize_weights();
		void reset_read();
		void close_file();

		int guess(char * buffer);
		void load_training();
		void reset_training();
		void load_buffer(char * buffer, bool random, int iteration);

};

#endif

/*
* Basic idea:
* C = argmax( vec_w x vec_x )
* vec_w = (alpha1, alpha2, ... , Beta)
* vec_w is arbritrarily chosen initially. (all 0 or randomize)
* vec_x = (f1, f2, ... , 1)
* f denotes if (X = x).
*
* During training:
* 1) initialize vec_w arbritarily
* 2) Go through the training set:
* 	- if Guess = actual label
*		- do nothing.
*	- else (Guess C, but actually its D)
* 		- vec_w(C) = vec_w(C) - eta*vec_w(C)
*		- vec_w(D) = vec_w(D) + eta*vec_w(D)
* 	- eta(n) = 1/n, where n denotes the nth epoch.
* 3) Go through enough epochs until it converges.
* 
*/

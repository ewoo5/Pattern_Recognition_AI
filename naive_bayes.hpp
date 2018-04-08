#include <iostream>
#include <fstream>
#include <vector>
#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

class naive_bayes{

	private:
		int width, height; // width: x-axis, lenght: y-axis, sizes for each picture.
		float **** probability_matrix; // stored as: probability[number class][row][column][0 - numerator; 1 - denominator].
		std::ifstream training_txt, testing_txt;

	public:
		naive_bayes();
		naive_bayes(int in_width, int in_height, char * training, char * testing);
		~naive_bayes();
		void train();
		void test();
		
		void close_file(); // to allow other things like the perceptron object to access the training/testing data.

};

#endif

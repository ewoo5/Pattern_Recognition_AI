#include <iostream>
#include <fstream>
#include <cstdlib>
#include "perceptron.hpp"

static int ** training = NULL;
static bool * visited = NULL;
static int training_size = 0;

perceptron::perceptron(){

	width = height = 0;
	vec_w = NULL;
	eta = 1;

}

perceptron::perceptron(int in_width, int in_height, char * in_train, char * in_test){

	width = in_width;
	height = in_height;
	training_txt.open(in_train, std::fstream::in);
	testing_txt.open(in_test, std::fstream::in);

	vec_w = new float***[10];
	int a, b, c;
	for(a = 0; a < 10; a++){
		vec_w[a] = new float**[height + 1];
		for(b = 0; b < height; b++){
			vec_w[a][b] = new float*[width];
			for(c = 0; c < width; c++){
				vec_w[a][b][c] = new float[2];
				vec_w[a][b][c][0] = 0;
				vec_w[a][b][c][1] = 0;
			}
		}
		vec_w[a][b] = new float*[1]; // For Beta in (w*x + b), so Beta will be in vec_w[number][height][0][0].
		vec_w[a][b][0] = new float[2];
		vec_w[a][b][0][0] = 0;
		vec_w[a][b][0][1] = 1;
	}

}

perceptron::~perceptron(){

	if(training_txt.is_open()){
		training_txt.close();
	}
	if(testing_txt.is_open()){
		testing_txt.close();
	}

	int a, b, c;
	for(a = 0; a < 10; a++){
		for(b = 0; b < height; b++){
			for(c = 0; c < width; c++){
				delete [] vec_w[a][b][c];
			}
			delete [] vec_w[a][b];
		}

		delete [] vec_w[a][height][0];
		delete [] vec_w[a][height];

		delete [] vec_w[a];
	}
	delete [] vec_w;

	for(a = 0 ; a < training_size; a++){
		delete [] training[a];
	}
	
	delete [] visited;

}

void perceptron::reset_read(){

	training_txt.clear();
	testing_txt.clear();
	training_txt.seekg(0, training_txt.beg);
	testing_txt.seekg(0, testing_txt.beg);

}

void perceptron::randomize_weights(){



}

void perceptron::train(int epochs){

	int n, i;
	char buffer[width*height + 2]; // stores it in the format: [x1,...,xn,Beta,number]
	
	load_training();

	for(n = 1; n <= epochs; n++){

		reset_read();
		int i;
		int row, column, x;
		int number;
		int current_guess;
		
		for(i = 0; i < training_size; i++){

			load_buffer(buffer, false, i);
			number = buffer[width*height + 1];
			//std::cout<<"Correct number: "<<number;
			current_guess = guess(buffer);
			//std::cout<<", Guess: "<<current_guess<<"\n";
			if(current_guess != number){
				for(row = 0; row < height; row++){
					for(column = 0; column < width; column++){
						vec_w[number][row][column][0] += (float)1/n * buffer[row*width + column];
						//std::cout<<vec_w[number][row][column][0]<<"\n";
						vec_w[current_guess][row][column][0] -= (float)1/n * buffer[row*width + column];
					}
				}
				vec_w[number][row][0][0] += (float)1/n * buffer[width*height];
				vec_w[current_guess][row][0][0] -= (float)1/n * buffer[width*height];
			}			

		}

		// Now test your accuracy and print your results.
		std::cout<<"\n*************Epoch "<<n<<"*************\n";
		test();

	}

}

/*
* Returns the bes guess with the current weights.
*/
int perceptron::guess(char * buffer){

	bool ret;
	float best_score = -999999;
	int best_class = 0;
	
	int i, row, column;
	float current_score;
	for(i = 0; i < 10; i++){
		current_score = 0;
		for(row = 0; row < height; row++){
			for(column = 0; column < width; column++){
				current_score += vec_w[i][row][column][0] * buffer[row*width + column];			
			}
		}
		current_score += vec_w[i][row][0][0] * buffer[height*width];
		
		//std::cout<<current_score<<"\n";
		if(current_score > best_score){
			best_class = i;
			best_score = current_score;
		}
	}

	return best_class;

}

/*
* Loads the whole training data into a buffer "training" declared at the top.
*/
void perceptron::load_training(){

	int block_size = width*height + height + 2 + 1;

	training_txt.seekg(0, training_txt.end);
	int file_size = training_txt.tellg();
	training_txt.seekg(0, training_txt.beg);

	training_size = (file_size + 1)/block_size;
	std::cout<<"Training size: "<<training_size<<"\n";
	
	training = new int*[training_size];
	int i, row, column;
	char buffer[block_size];
	for(i = 0; i < training_size; i++){
		training[i] = new int[width*height + 2];
		training_txt.read(buffer, block_size);
		for(row = 0; row < height; row++){
			for(column = 0; column < width; column++){
				training[i][row*width + column] = buffer[row*width + column + row] - '0';
			}
		}
		training[i][width*height] = 1;
		training[i][width*height + 1] = buffer[block_size - 2] - '0'; // This last element store the class/number.
	}

	visited = new bool[training_size];
	for(i = 0; i < training_size; i++){
		visited[i] = false;
	}

}


/*
* Resets all the training data to unvisited.
* This is to prepare for the next epoch run through.
*/
void perceptron::reset_training(){

	int i;
	for(i = 0; i < training_size; i++){
		visited[i] = false;
	}

}

/*
* Given the input buffer.
* Fill it with the appropriate training data.
* if random = TRUE, it will pick a training data randomly.
* if random = FALSE, it will read the training dataset sequencially.
*
* NOTE: buffer size must be width*height + 1.
*/
void perceptron::load_buffer(char * buffer, bool random, int iteration){

	int i;
	if(iteration == 0){
		reset_training();
	}	

	if(random){
		i = std::rand() % training_size;
		for(i = 0; i < width*height + 2; i++){
			buffer[i] = training[iteration][i];
		}
	}else{
		for(i = 0; i < width*height + 2; i++){
			buffer[i] = training[iteration][i];
		}
	}

}

void perceptron::test(){

	int i, row, column;
	float best_score = -9999999999;
	float current_score;
	int best_guess = 0;

	int accuracy[10][2]; // accuracy[class][0-correct; 1-tokens]
	for(i = 0; i < 10; i++){
		accuracy[i][0] = 0;
		accuracy[i][1] = 1;
	}

	int block_size = width*height + height + 1 + 2;
	char buffer[block_size];
	int correct_number;
	
	testing_txt.read(buffer, block_size);
	while(!testing_txt.eof() && !testing_txt.fail()){
		
		best_score = -9999999999;
		correct_number = buffer[block_size - 2] - '0';
		//std::cout<<"Correct number: "<<correct_number;
		for(i = 0; i < 10; i++){
			current_score = 0;
			for(row = 0; row < height; row++){
				for(column = 0; column < width; column++){
					current_score += vec_w[i][row][column][0] * (buffer[row*width + column + row] - '0');
				}
			}
			current_score += vec_w[i][row][0][0];
			if(current_score > best_score){
				best_score = current_score;
				best_guess = i;
			}
		}

		//std::cout<<", Guess: "<<best_guess<<"\n";
		if(best_guess == correct_number){
			accuracy[correct_number][0] += 1;
		}
		accuracy[correct_number][1] += 1;	
	
		testing_txt.read(buffer, block_size);
	}

	// Repeat once more because the last one has size 1 less than block size.
	correct_number = buffer[block_size - 2] - '0';
	for(i = 0; i < 10; i++){
		current_score = 0;
		for(row = 0; row < height; row++){
			for(column = 0; column < width; column++){
				current_score += vec_w[i][row][column][0] * (buffer[row*width + column + row] - '0');
			}
		}
		current_score += vec_w[i][row][0][0];
		if(current_score > best_score){
			best_score = current_score;
			best_guess = i;
		}
	}

	if(best_guess == correct_number){
		accuracy[correct_number][0] += 1;
	}
	accuracy[correct_number][1] += 1;	

	// print accuracy results.
	float percen;
	int nume = 0, denom = 0;
	for(i = 0; i < 10; i++){
		percen = (float)accuracy[i][0]/accuracy[i][1];
		nume += accuracy[i][0];
		denom += accuracy[i][1];
		std::cout<<"Class: "<<i<<", fit: "<<percen<<"\n";
	}
	float overall = (float)nume/denom;
	std::cout<<"Overall accuracy: "<<overall<<"\n";
	
}





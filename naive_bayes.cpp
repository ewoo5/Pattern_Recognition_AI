#include <iostream>
#include <fstream>
#include <cmath>
#include "naive_bayes.hpp"

#define V_OFF 0.1

naive_bayes::naive_bayes(){

	height = width = 0;
	probability_matrix = NULL;	

}

naive_bayes::naive_bayes(int in_width, int in_height, char * training, char * testing){

	width = in_width;
	height = in_height;
	training_txt.open(training, std::fstream::in);
	testing_txt.open(testing, std::fstream::in);

	// Hard code, assume this can only categorize/classify pictures of numbers 0 - 9.
	probability_matrix = new float***[10];
	int a, b, c;
	for(a = 0; a < 10; a++){
		probability_matrix[a] = new float**[height];
		for(b = 0; b < height; b++){
			probability_matrix[a][b] = new float*[width];
			for(c = 0; c < width; c++){
				probability_matrix[a][b][c] = new float[2];
				probability_matrix[a][b][c][0] = V_OFF; // numerator + 1 for laplace smoothing.
				probability_matrix[a][b][c][1] = 2*V_OFF; // denominator + 2 for laplace smoothing. (+1 for 0-case and +1 for 1-case).
			}
		}
	}

}

naive_bayes::~naive_bayes(){

	if(training_txt.is_open()){
		training_txt.close();
	}
	if(testing_txt.is_open()){
		testing_txt.close();
	}

	int a, b, c;
	if(probability_matrix != NULL){
		for(a  = 0; a < 10; a++){
			if(probability_matrix[a] != NULL){
				for(b = 0; b < height; b++){
					if(probability_matrix[a][b] != NULL){
						for(c = 0; c < width; c++){
							if(probability_matrix[a][b][c] != NULL){
								delete [] probability_matrix[a][b][c];
							}
						}
						delete [] probability_matrix[a][b];
					}
				}
				delete [] probability_matrix[a];
			}
		}
		delete [] probability_matrix;
	}

}

/*
* tallies all the data for calculating the probabilities.
*/
void naive_bayes::train(){

	// a box/position is written over if it is "1", and "0" if it is just the background.
	int current_class, row, column, count = 0;
	int block_size = width*height + 2 + height + 1; // add 2 for the class label at the end
	char buffer[block_size]; // add height because there will be '\n' at the end of each row.

	training_txt.read(buffer, block_size);
	int i = 0, j;
	while(!training_txt.eof() && !training_txt.fail()){
		count += 1;
		current_class = (int)(buffer[block_size - 2] - '0'); // class label is stored in the last-most character.
		for(row = 0; row < height; row++){
			for(column = 0; column < width; column++){
				probability_matrix[current_class][row][column][0] += (buffer[row*width + column + row] - '0'); // + row because of the '\n' character.
				probability_matrix[current_class][row][column][1] += 1;
			}
		}
		
		training_txt.read(buffer, block_size);
	}

	count += 1;
	current_class = (int)(buffer[block_size - 2] - '0'); // class label is stored in the last-most character.
	for(row = 0; row < height; row++){
		for(column = 0; column < width; column++){
			probability_matrix[current_class][row][column][0] += (buffer[row*width + column + row] - '0'); // + row because of the '\n' character.
			probability_matrix[current_class][row][column][1] += 1;
		}
	}
	
	training_txt.read(buffer, block_size);

	std::cout<<"number of training samples: "<<count<<"\n";

	int left, right;
	left = 2;
	right = 8;
	std::cout<<"\nLikelihood map for "<<left<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for "<<right<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for Odds(Fij = 1, "<<left<<", "<<right<<"):\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = std::log((probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1])/(probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1]));
			if(likelihood > -0.5){
				if(likelihood > 0.5){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	left = 3;
	right = 9;
	std::cout<<"\nLikelihood map for "<<left<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for "<<right<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for Odds(Fij = 1, "<<left<<", "<<right<<"):\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = std::log((probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1])/(probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1]));
			if(likelihood > -0.5){
				if(likelihood > 0.5){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}
	
	left = 9;
	right = 5;
	std::cout<<"\nLikelihood map for "<<left<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for "<<right<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for Odds(Fij = 1, "<<left<<", "<<right<<"):\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = std::log((probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1])/(probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1]));
			if(likelihood > -0.5){
				if(likelihood > 0.5){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	left = 4;
	right = 7;
	std::cout<<"\nLikelihood map for "<<left<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for "<<right<<":\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = (float)probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1];
			if(likelihood > 0.5){
				if(likelihood > 0.75){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

	std::cout<<"\nLikelihood map for Odds(Fij = 1, "<<left<<", "<<right<<"):\n";
	for(row = 0; row < 32; row++){
		for(column = 0; column < 32; column++){
			float likelihood = std::log((probability_matrix[left][row][column][0]/probability_matrix[left][row][column][1])/(probability_matrix[right][row][column][0]/probability_matrix[right][row][column][1]));
			if(likelihood > -0.5){
				if(likelihood > 0.5){
					std::cout<<" ";
				}else{
					std::cout<<"+";
				}
			}else{
				std::cout<<"-";
			}
		}
		std::cout<<"\n";
	}

}

/*
* goes through the testing set and evaluates how well it's predictions are.
*/
void naive_bayes::test(){

	int performance_matrix[10][2]; // performance_matrix[class label][0-numerator; 1-denominator]
	int confusion_matrix[10][10]; // [correct number][guessed number], value is the number times that it is in that guess.
	int posterior_matrix[10][2][2]; // [class][0-best; 1-worst][0-best posterior; 1-token number]
	int i, j, token = 0;
	int overall_correct = 0, total = 0;
	for(i = 0; i < 10; i++){
		performance_matrix[i][0] = 0;
		performance_matrix[i][1] = 0;
	}
	for(i = 0; i < 10; i++){
		for(j = 0; j < 10; j++){
			confusion_matrix[i][j] = 0;
		}
	}
	for(i = 0; i < 10; i++){
		posterior_matrix[i][0][0] = -9999999;
		posterior_matrix[i][0][1] = 0;
		posterior_matrix[i][1][0] = 9999999;
		posterior_matrix[i][1][1] = 0;
	}	

	int correct_class, best_guess, row, column, block_size;
	float best_prob;
	block_size = height*width + height + 2 + 1; //add 2 for the class label at the end, add height because there will be '\n' at the end of each row.
		
	char buffer[block_size];
	
	testing_txt.read(buffer, block_size);
	while(!testing_txt.eof() && !testing_txt.fail()){
		
		best_prob = -10000000;
		correct_class = buffer[block_size - 2] - '0';

		float prob;
		for(i = 0; i < 10; i++){ // loop through all possible number/classes to see which has the best probability (MAP - maximum a posteriori).
			prob = 0;
			int denom = 0;
			for(j = 0; j < 10; j++){
				denom += probability_matrix[j][1][1][1];
			}
			prob += std::log(probability_matrix[i][1][1][1]) - std::log(denom); // this is P(C = c)
			for(row = 0; row < height; row++){
				for(column = 0; column < width; column++){
					//this is += x*(log[P(X = 1|C = c)] - log[1 - P(X = 1|C = c)]) = x*log[# of (X = 1 ^ C = c)/(# of c - # of (X = 1 ^ C = c))];
					prob += (int)(buffer[row*(width + 1) + column] - '0')*(std::log(probability_matrix[i][row][column][0]) - std::log(probability_matrix[i][row][column][1] - probability_matrix[i][row][column][0]));
					//this is += log[1 - P(X = 1|C = c)]
					prob += std::log(probability_matrix[i][row][column][1] - probability_matrix[i][row][column][0]) - std::log(probability_matrix[i][row][column][1]);
				}
			}
			if(prob > best_prob){
				best_prob = prob;
				best_guess = i;
			}

		}

		performance_matrix[correct_class][1] += 1;
		total += 1;
		if(best_guess == correct_class){
			performance_matrix[correct_class][0] += 1;
			overall_correct += 1;
		}
		
		confusion_matrix[correct_class][best_guess] += 1;
		
		if(best_prob > posterior_matrix[correct_class][0][0]){
			posterior_matrix[correct_class][0][0] = best_prob;
			posterior_matrix[correct_class][0][1] = token;
		}

		if(best_prob < posterior_matrix[correct_class][1][0]){
			posterior_matrix[correct_class][1][0] = best_prob;
			posterior_matrix[correct_class][1][1] = token;
		}
			
		testing_txt.read(buffer, block_size);
		token++;

	}

	// Repeat one last time
	best_prob = -10000000;
	correct_class = buffer[block_size - 2] - '0';

	float prob;
	for(i = 0; i < 10; i++){ // loop through all possible number/classes to see which has the best probability (MAP - maximum a posteriori).
		prob = 0;
		int denom = 0;
		for(j = 0; j < 10; j++){
			denom += probability_matrix[j][1][1][1];
		}
		prob += std::log(probability_matrix[i][1][1][1]) - std::log(denom); // this is P(C = c)
		for(row = 0; row < height; row++){
			for(column = 0; column < width; column++){
				//this is += x*(log[P(X = 1|C = c)] - log[1 - P(X = 1|C = c)]) = x*log[# of (X = 1 ^ C = c)/(# of c - # of (X = 1 ^ C = c))];
				prob += (int)(buffer[row*(width + 1) + column] - '0')*(std::log(probability_matrix[i][row][column][0]) - std::log(probability_matrix[i][row][column][1] - probability_matrix[i][row][column][0]));
				//this is += log[1 - P(X = 1|C = c)]
				prob += std::log(probability_matrix[i][row][column][1] - probability_matrix[i][row][column][0]) - std::log(probability_matrix[i][row][column][1]);
			}
		}
		if(prob > best_prob){
			best_prob = prob;
			best_guess = i;
		}
	}
	
	performance_matrix[correct_class][1] += 1;
	total += 1;
	if(best_guess == correct_class){
		performance_matrix[correct_class][0] += 1;
		overall_correct += 1;
	}
	
	confusion_matrix[correct_class][best_guess] += 1;
		
	if(best_prob > posterior_matrix[correct_class][0][0]){
		posterior_matrix[correct_class][0][0] = best_prob;
		posterior_matrix[correct_class][0][1] = token;
	}

	if(best_prob < posterior_matrix[correct_class][1][0]){
		posterior_matrix[correct_class][1][0] = best_prob;
		posterior_matrix[correct_class][1][1] = token;
	}
	
	for(i = 0; i < 10; i++){
		std::cout<<"class: "<<i<<", fit: "<<(float)performance_matrix[i][0]/(float)performance_matrix[i][1]<<"\n";
	}
	std::cout<<"Overall acuracy: "<<(float)overall_correct/(float)total<<"\n";

	std::cout<<"\n*************************Confusion Matrix************************\n";
	std::cout<<"\t\t\t\t\tGuess\t\t\t\t\t\n\t\t";
	for(i = 0; i < 10; i++){
		std::cout<<i<<":\t";
	}
	std::cout<<"\n\nClass";
	for(i = 0; i < 10; i++){
		std::cout<<"\t"<<i<<":";
		float guess_percen;
		int class_total = 0;
		for(j = 0; j < 10; j++){
			class_total += confusion_matrix[i][j];
		}
		for(j = 0; j < 10;j++){
			guess_percen = (float) confusion_matrix[i][j]/class_total;
			std::cout<<"\t";
			//std::cout.precision(3);
			std::cout<<guess_percen;
		}
		std::cout<<"\n";
	}

	training_txt.clear();
	testing_txt.clear();
	training_txt.seekg(0, training_txt.beg);
	testing_txt.seekg(0, testing_txt.beg);

	std::cout<<"\n";
	for(i = 0; i < 10; i++){
		std::cout<<"Highest Posterior for "<<i<<", with value "<<posterior_matrix[i][0][0]<<":\n";
		testing_txt.seekg(posterior_matrix[i][0][1]*block_size, testing_txt.beg);
		testing_txt.read(buffer, block_size);
		for(j = 0; j < block_size; j++){
			std::cout<<buffer[j];
		}
		
		std::cout<<"Lowest Posterior for "<<i<<", with value "<<posterior_matrix[i][1][0]<<":\n";
		testing_txt.seekg(posterior_matrix[i][1][1]*block_size, testing_txt.beg);
		testing_txt.read(buffer, block_size);
		for(j = 0; j < block_size; j++){
			std::cout<<buffer[j];
		}
		std::cout<<"\n";
	}
	
}

void naive_bayes::close_file(){

	if(training_txt.is_open()){
		training_txt.close();
	}
	if(testing_txt.is_open()){
		testing_txt.close();
	}

}



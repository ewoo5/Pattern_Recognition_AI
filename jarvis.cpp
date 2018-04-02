#include <iostream>
#include <fstream>
#include "naive_bayes.hpp"

using namespace std;

int main(int argc, char * argv[]){

	if(argc != 5){
		cout<<"Please execute in the following format:\n ./main <height> <width> <training.txt> <testing.txt>\n";
		return 0;
	}

	int width = (int)(argv[2][1] - '0' + 10*(argv[2][0] - '0'));
	cout<<"width = "<<width<<"\n";
	int height = (int)(argv[1][1] - '0' + 10*(argv[1][0] - '0'));
	cout<<"height = "<<height<<"\n";
	naive_bayes brain(width, height, argv[3], argv[4]);
	
	brain.train();
	brain.test();
	
	return 0;

}

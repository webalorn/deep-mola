#include "headers/layer_network.hpp"
#include <string>
#include <iostream>
#include <fstream>

const int ENTRY_SIZE_1 = 784;
const int ENTRY_SIZE_2 = 10;

DataSet readDatasFrom(std::string filename) {
	std::ifstream infile(filename);
	int nbEntries;
	infile >> nbEntries;

	DataSet datas(nbEntries);
	for (int iEntry = 0; iEntry < nbEntries; iEntry++) {
		datas[iEntry].first.resize(ENTRY_SIZE_1);
		datas[iEntry].second.resize(ENTRY_SIZE_2);

		for (real& v : datas[iEntry].first) {
			infile >> v;
		}
		for (real& v : datas[iEntry].second) {
			infile >> v;
		}
	}
	return datas;
}

int main() {
	std::ios::sync_with_stdio(false);

	std::cerr << "Read datas...\n";

	//*
	auto trainingData = readDatasFrom("datas/training.in");
	auto validationData = readDatasFrom("datas/validation.in");
	auto testData = readDatasFrom("datas/test.in");//*/

	/*
	auto trainingData = readDatasFrom("datas/test.in");
	auto testData = trainingData;//*/

	std::cerr << "Start training\n";

	LayerNetwork network({784, 100, 10});
	network.trainSGD(trainingData, 30, 10, 3.0, testData);
}
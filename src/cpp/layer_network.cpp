#include "../headers/layer_network.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

real sigmoid(real z) {
	return 1.0 / (1.0 + std::exp(-z));
}
Vect sigmoid(Vect z) {
	return vectApply(z, &sigmoid);
}

real sigmoidPrime(real z) {
	return sigmoid(z) * (1 - sigmoid(z));
}
Vect sigmoidPrime(Vect z) {
	return vectApply(z, &sigmoidPrime);
}

NetState::NetState(int nbLayers) {
	neuronIn.resize(nbLayers);
	neuronOut.resize(nbLayers);
}

LayerNetwork::LayerNetwork(std::vector<int> p_layerSizes) {
	nbLayers = p_layerSizes.size();
	layersSizes = p_layerSizes;
	
	// We create adges between neurons

	std::default_random_engine generator;
	std::normal_distribution<real> gaussDistrib(0, 1);

	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
		Matrix edges = std::vector<std::vector<real>>(layersSizes[iLayer+1], // constant neuron doesn't have inputs
			std::vector<real>(layersSizes[iLayer] + 1, 0) // +1 for the constant neuron
		);
		for (auto& vect : edges) {
			for (real& weight : vect) {
				weight = gaussDistrib(generator);
			}
		}
		weights.push_back(edges);
	}
	
}

NetState LayerNetwork::feedForward(Vect input) {
	// First neuron layer is a "fake" layer, neurons represent the input
	NetState state(nbLayers);
	state.neuronIn[0] = input;
	state.neuronOut[0] = input;

	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) { // From iLayer to iLayer + 1
		state.neuronOut[iLayer].push_back(1); // Fake neuron with constant output for bias
		state.neuronIn[iLayer + 1] = weights[iLayer] * state.neuronOut[iLayer];
		state.neuronOut[iLayer + 1] = sigmoid(state.neuronIn[iLayer + 1]);
	}
	return state;
}

Vect LayerNetwork::quickFeedForward(Vect input) {
	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
		input.push_back(1); // Fake neuron with constant output for bias
		input = sigmoid(weights[iLayer] * input);
	}
	return input;
}

void LayerNetwork::trainSGD(DataSet trainingData, int epochs, int batchSize, real learningRate, DataSet testData) {
	int nbExamples = trainingData.size();

	for (int iEpoch = 0; iEpoch < epochs; iEpoch++) {
		std::random_shuffle(trainingData.begin(), trainingData.end());
		std::vector<DataSet> batches(nbExamples/batchSize);
		for (uint iBatch = 0; iBatch < batches.size(); iBatch++) {
			for (uint iExample = iBatch * batchSize; iExample < (iBatch + 1) * batchSize; iExample++) {
				batches[iBatch].push_back(trainingData[iExample]);
			}
		}

		for (DataSet& batch : batches) {
			trainWithBatch(batch, learningRate);
		}

		std::cerr << "Epoch " << iEpoch << " complete\n";
		if (testData.size()) {
			int nbSuccess = evaluateClassify(testData);
			real successRate = (real)nbSuccess / (real)testData.size() * 100;
			std::cerr << "Success rate: " << successRate << "% (" << nbSuccess << " over " << testData.size() << ")\n";
		}
	}
}

void LayerNetwork::trainWithBatch(DataSet& batch, real learningRate) {
	std::vector<Matrix> weightsGrad(nbLayers-1);
	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
		weightsGrad[iLayer] = newMatrix(weights[iLayer].size(), weights[iLayer][0].size(), 0);
	}

	for (std::pair<Vect, Vect>& example : batch) {
		Vect& input = example.first;
		Vect& output = example.second;
		std::vector<Matrix> weightDeltas = backPropagation(input, output);

		for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
			weightsGrad[iLayer] = weightsGrad[iLayer] + weightDeltas[iLayer];
		}
	}

	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
		weights[iLayer] = weights[iLayer] + ( weightsGrad[iLayer] * ( -1.0 * learningRate / (real)batch.size() ) );
	}
}

std::vector<Matrix> LayerNetwork::backPropagation(Vect input, Vect output) {
	NetState network = feedForward(input);

	std::vector<Vect> neuronDelta(nbLayers);

	neuronDelta[nbLayers-1] = (network.neuronOut.back() - output) * sigmoidPrime(network.neuronIn.back());

	for (int iLayer = nbLayers - 2; iLayer > 0; iLayer--) {
		Vect backProg = (transpose(weights[iLayer]) * neuronDelta[iLayer+1]);
		backProg.pop_back(); // We don't want to compute delta for the constant neuron
		neuronDelta[iLayer] = backProg * sigmoidPrime(network.neuronIn[iLayer]);
	}

	std::vector<Matrix> weightDeltas(nbLayers-1);
	for (int iLayer = 0; iLayer < nbLayers-1; iLayer++) {
		weightDeltas[iLayer] = newMatrix(layersSizes[iLayer+1], layersSizes[iLayer] + 1);
		for (int neuron1 = 0; neuron1 <= layersSizes[iLayer]; neuron1++) { // +1 for the constant neuron
			for (int neuron2 = 0; neuron2 < layersSizes[iLayer+1]; neuron2++) {
				weightDeltas[iLayer][neuron2][neuron1] = network.neuronOut[iLayer][neuron1] * neuronDelta[iLayer+1][neuron2];
			}
		}
	}
	return weightDeltas;
}

int LayerNetwork::evaluateClassify(DataSet& testData) {
	int nbSuccess = 0;
	for (std::pair<Vect, Vect>& test : testData) {
		Vect result = quickFeedForward(test.first);
		int id1 = std::distance(test.second.begin(), max_element(test.second.begin(), test.second.end()));
		int id2 = std::distance(result.begin(), max_element(result.begin(), result.end()));
		if (id1 == id2) {
			nbSuccess += 1;
		}
	}
	return nbSuccess;
}
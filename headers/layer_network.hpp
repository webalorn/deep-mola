#ifndef LAYER_NETWORK_HPP_
#define LAYER_NETWORK_HPP_

#include "types.hpp"

real sigmoid(real);
Vect sigmoid(Vect);

real sigmoidPrime(real);
Vect sigmoidPrime(Vect);

struct NetState {
	std::vector<Vect> neuronIn, neuronOut;
	NetState(int nbLayers);
};

class LayerNetwork {
	/* The network consider that a neuron with an input always equal to 1 is added at the end of each layer,
		and the first weight to each layer neuron is the weight between this neuron and the constant neuron.
		This weight is the bias */

private:
	int nbLayers;
	std::vector<int> layersSizes;
	std::vector<Matrix> weights; /* weights[i][j][k] is the weight between the j-th neuron of the layer i+1
								    and the k-th neuron of layer i */

	void trainWithBatch(DataSet&, real learningRate);
	std::vector<Matrix> backPropagation(Vect input, Vect output);

public:
	LayerNetwork(std::vector<int> p_layerSizes);
	NetState feedForward(Vect input); // Return all network state, may be slower than 'quickFeedForward'
	Vect quickFeedForward(Vect input); // Only return output

	// Stochastic Gradient Descent
	void trainSGD(DataSet trainingData, // List of pair of input/output
		int epochs, // Number of iterations of the main training loop
		int batchSize, // Size of the set of training example took for training each "turn"
		real learningRate,
		DataSet testData={} // To evaluate network accuracy at each step
	);
	int evaluateClassify(DataSet&); // Evaluate if the network can classify examples (if the maximum is at the right position)
};

#endif  /* !LAYER_NETWORK_HPP_ */
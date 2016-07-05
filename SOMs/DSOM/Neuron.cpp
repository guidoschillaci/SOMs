/*
 * This is an implementation of the Dynamic DSOM algorithm
 * Nicolas Rougier and Yann Boniface, "Dynamic Self-Organising Map", Neurocomputing 74, 11 (2011), pp. 1840-1847
 *
 * DSOM algorithm is essentially a variation of the DSOM algorithm where the time dependency has been removed.
 *
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
 *
*/

#include "Neuron.h"

namespace dsom
{

    void Neuron::init(int x, int y, SOMs::ParamSOM* _params)
    {
        // set GenericNeuron position in the DSOM grid
        X = x;
        Y = y;
        params = _params;
        // initialize weights
        weights.resize(params->weights_size);
        for (int k = 0; k < params->weights_size; k++)
            weights[k] = ((float) rand() / (RAND_MAX)); // between 0 and 1

    }

    void Neuron::init(int x, int y, SOMs::ParamSOM* _params, std::vector<float> *min, std::vector<float> *max)
    {        
        // set GenericNeuron position in the DSOM grid
        X = x;
        Y = y;
        params = _params;

        // initialize weights
        weights.resize(params->weights_size);

        for (int k = 0; k < params->weights_size; k++)
        {
            weights[k] = (((float) rand() / (RAND_MAX)) * (max->at(k)-min->at(k))) + min->at(k); // between min and max
        }
    }

    void Neuron::init(int x, int y, SOMs::ParamSOM* _params, std::vector<float> *min, std::vector<float> *max, std::vector<float> *means, std::vector<float> *stddev)
    {
        // set GenericNeuron position in the DSOM grid
        X = x;
        Y = y;
        params = _params;

        // initialize weights
        weights.resize(params->weights_size);

        for (int k = 0; k < params->weights_size; k++)
        {
            weights[k] = Utils::normal(means->at(k),stddev->at(k));
        }
    }

    // this is the function that adjust the weights of the neuron according to its distance to the
    // best matching unit (winner neuron)
    float Neuron::updateWeights(std::vector<float> *input_pattern, float neighborhood_function, int iteration, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        if (weights.size()==0)
            std::cerr<<"neuron:updateWeights - weight size == 0 "<<std::endl;
        if (input_pattern->size()==0)
            std::cerr<<"neuron:updateWeights - input_pattern size == 0 "<<std::endl;
        float sum = 0;
        float delta = 0;
        float distance = getDistance(input_pattern, features_min, features_max);
        for (unsigned int i = 0; i < weights.size(); i++)
        {
            delta = params->DSOM_learning_rate * distance * neighborhood_function * (input_pattern->at(i) - weights[i]);
            weights[i] += delta;
            sum += delta;
        }
        return sum / float(weights.size());
    }
}


/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#include "Neuron.h"

namespace classicSOM
{

    /**
     * @brief Neuron::updateWeights
     * This implements the function for the weights update of the classical self-organising map algorithm
     * It adjusts the weights of the neuron according to its distance to the best matching unit (winner neuron)
     * The statistics of the dimensions in the weights are not actually used.
     * @param input_pattern the input pattern
     * @param learning_rate this value decrease over time
     * @param neighborhood_function this value decrease over time
     * @param features_min a vector containing the minimum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param features_max a vector containing the maximum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @return
     */
    float Neuron::updateWeights(std::vector<float> *input_pattern, float learning_rate, float neighborhood_function, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        if (weights.size()==0)
            std::cerr<<"neuron:updateWeights - weight size == 0 "<<std::endl;
        if (input_pattern->size()==0)
            std::cerr<<"neuron:updateWeights - input_pattern size == 0 "<<std::endl;
        float sum = 0;
        float delta = 0;
        for (unsigned int i = 0; i < weights.size(); i++)
        {
            delta = learning_rate * neighborhood_function * (input_pattern->at(i) - weights[i]);
            weights[i] += delta;
            sum += delta;
        }
        return sum / float(weights.size());
    }


}


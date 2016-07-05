/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/

#ifndef ClassicSOM_NEURON_H
#define ClassicSOM_NEURON_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <boost/random/normal_distribution.hpp>
#include "SOMs/GenericNeuron.h"

namespace classicSOM
{

    /**
     * @brief The Neuron class of the standard Kohonen SOM algorithm.
     *
     * It implements the update function of the standard Kohonen SOM algorithm.
     * After a number of iterations, the parameters learning_rate and sigma are
     * annealed exponentially.
     */
    class Neuron : public SOMs::GenericNeuron
    {
    public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            Neuron() : SOMs::GenericNeuron() { }
            Neuron(int x, int y, SOMs::ParamSOM* _params) : SOMs::GenericNeuron( x, y, _params) { }
            Neuron(const SOMs::GenericNeuron& n) : SOMs::GenericNeuron(n) { }

            ////////////////////
            // METHODS OVERRIDED

            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS

            // this is the function that adjust the weights of the neuron according to its distance to the
            // best matching unit (winner neuron)
            float updateWeights(std::vector<float> *input_pattern, float learning_rate, float neighborhood_function, std::vector<float> *features_min, std::vector<float> *features_max);

            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS
            // none
    };
}

#endif // ClassicSOM_NEURON_H

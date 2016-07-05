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

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <boost/random/normal_distribution.hpp>
#include "SOMs/GenericNeuron.h"

//#define ACTIVATION_FUNCTION_TANH    0
//#define ACTIVATION_FUNCTION_MEX_HAT 1

namespace dsom
{

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
            Neuron(int x, int y, SOMs::ParamSOM *param) : SOMs::GenericNeuron( x, y, param) { }
//            Neuron(int x, int y, int siz, int weights_size);
            Neuron(const SOMs::GenericNeuron& n) : SOMs::GenericNeuron(n)
            {
                std::cout<<"called Neuron(const SOMs::GenericNeuron& n). size"<<weights.size()<<std::endl;
            }


            ////////////////////
            // METHODS OVERRIDED
            void init(int x, int y, SOMs::ParamSOM* _params);
            void init(int x, int y, SOMs::ParamSOM* _params, std::vector<float> *min, std::vector<float> *max);
            void init(int x, int y, SOMs::ParamSOM* _params, std::vector<float> *min, std::vector<float> *max, std::vector<float> *mean, std::vector<float> *stddev);


            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS

            // this is the function that adjust the weights of the neuron according to its distance to the
            // best matching unit (winner neuron)
            float updateWeights(std::vector<float> *input_pattern, float neighborhood_function, int iteration, std::vector<float> *features_min, std::vector<float> *features_max);

            // Used in training the Hebbian tables
            // Running_mean and running_std_dev are used for normalising the data for calculating the distance
            //float getActivation (std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max)
//            float getActivation (std::vector<float> *input_pattern, std::vector<RunningStatistics> *running_avg_and_stddev);
//            float getActivation (std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max);

            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS
            // none
    };
}

#endif // NEURON_H

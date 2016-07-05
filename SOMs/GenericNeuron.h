/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#ifndef GenericNeuron_H
#define GenericNeuron_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <boost/random/normal_distribution.hpp>
#include "Utils/Common.h"

#include "SOMs/ParamSOM.h"


namespace SOMs
{

     /**
     * @brief The GenericNeuron class implements a generic node of a Self-Organising Map
     *
     * The GenericNeuron class implements a generic node of a Self-Organising Map.
     * The neuron contains a set of attributes such as its position in the SOM lattice
     * or the weights in the feature space, and a set of methods such as getDistance or
     * getActivation, used for propagating signals through Hebbian Tables between SOMs.
     *
     */
    class GenericNeuron
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            ParamSOM* params;
            std::vector<float> weights;

            int X; // GenericNeuron x index in the SOM lattice
            int Y; // GenericNeuron y index in the SOM lattice

            // gather all the incoming activations propagated to this node
            float propagated_activation;

            // how often it has been visited in learning the hebbian table
            float number_of_visits;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            GenericNeuron()
            {
                propagated_activation=0;
            }

            GenericNeuron(int x, int y, ParamSOM* _params)
            {
                params = _params;
                init(x, y, params);
            }

            GenericNeuron(const GenericNeuron& n)
            {
                params = n.params;
                propagated_activation=0;
                X = n.X;
                Y = n.Y;
                number_of_visits=n.number_of_visits;
                weights.resize(n.weights.size());
                for (int k = 0; k < n.weights.size(); k++)
                    weights[k] = n.weights[k];
            }

            GenericNeuron& operator=(GenericNeuron const &gn);

            ////////////////////
            // VIRTUAL FUNCTIONS

            // this is the function that adjust the weights of the GenericNeuron according to its distance to the
            // best matching unit (winner GenericNeuron)
            virtual float updateWeights(std::vector<float> *input_pattern, float neighborhood_function, int iteration,  std::vector<float> *features_min, std::vector<float> *features_max)
                { std::cerr<<"Calling GenericNeuron::updateWeights"<<std::endl; return 0; }
            virtual float updateWeights(std::vector<float> *input_pattern, float scaling_factor, float neighborhood_function, std::vector<float> *features_min, std::vector<float> *features_max)
                { std::cerr<<"Calling GenericNeuron::updateWeights"<<std::endl; return 0; }

            // Used in training the Hebbian tables
            float getActivation (std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max);

            ////////////////////
            // OTHER METHODS
            void init(int x, int y, ParamSOM* _param);
            void init(int x, int y, ParamSOM* _param, std::vector<float> *min, std::vector<float> *max);
            void init(int x, int y, ParamSOM* _param, std::vector<float> *min, std::vector<float> *max, std::vector<float> *means, std::vector<float> *stddev);

            float getDistance(std::vector<float> *patt1, std::vector<float> *patt2, std::vector<float> *features_min, std::vector<float> *features_max);
            float getDistance(std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max);
            float getSquaredDistance(std::vector<float> *patt1, std::vector<float> *patt2, std::vector<float> *features_min, std::vector<float> *features_max);
            float getSquaredDistance(std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max);

            bool weightsEqualTo(std::vector<float>* input);

            void printWeights();

    };
}

#endif // GenericNeuron_H

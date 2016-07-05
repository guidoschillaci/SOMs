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

#ifndef SOM_H
#define SOM_H

#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>

#include "SOMs/GenericSOM.h"
#include "SOMs/GenericNeuron.h"
#include "SOMs/DSOM/Neuron.h"

//#define ID_SENSORY_HAND_POSITION        0
//#define ID_SENSORY_LEFT_ARM_JOINTS      1
//#define ID_SENSORY_HEAD_JOINTS          2
//#define ID_MOTOR_LEFT_ARM_JOINTS_VEL    3
//#define ID_MOTOR_HEAD_JOINTS_VEL        4


namespace dsom
{

    class DSOM : public SOMs::GenericSOM
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////

            std::vector<float> learning_rate_history;
            std::vector<float> elasticity_history;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            DSOM(std::string filename);
            // dim = dimension of the weights of the neurons
            // siz = size of the DSOM (siz x siz)
            DSOM(SOMs::ParamSOM* _params);
            DSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max);
            DSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *means, std::vector<float> *stddevs);
            DSOM(const SOMs::GenericSOM& dsom);

            ////////////////////
            // METHODS OVERRIDED

            void init();
            void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max);
            void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *means, std::vector<float> *stddev);

            void loadFromFile (std::string filename);
            void saveNodes (std::string filename);
            void savePredictionErrors(std::string filename);
            void loadPredictionErrors(std::string filename);

            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS

            // Train step (only one input vector)
            float trainStep(std::vector<float> *input_pattern);
            float neighborhood_function(std::vector<float> *input_pattern, SOMs::GenericNeuron * winner, SOMs::GenericNeuron *current);

            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS

            void updateLearningRateAndElasticity();


    };

}

#endif // SOM_H

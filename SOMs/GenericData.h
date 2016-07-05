/*
 * This stores an input data and its normalised version
 *
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
 *
*/

#ifndef GenericData_H
#define GenericData_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <boost/random/normal_distribution.hpp>
#include "Utils/RunningStat.h"
#include "Utils/NaoTH/Common.h"

// normalisation of the input for the neuron activation function
#define NORMALISE_ACTIV_WITH_FEAT_SCALING       0 // scale the distance for calculating the activation between 0 and 1
#define NORMALISE_ACTIV_WITH_RUNNING_AVG_STDDEV 1
//#define NORMALISE_ACTIV_WITH_SOFT_MAX           2


namespace SOMs
{

    class GenericData
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////

            std::vector<float> data;
            std::vector<float> normalised_data;

            bool is_normalised;
            int normalisation_type;

            std::vector<float> *features_min;
            std::vector<float> *features_max;
            std::vector<RunningStat> *running_avg_and_stddev;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            GenericData() {}
            GenericData(std::vector<float> *in);
            GenericData(std::vector<float> *in, int _normalisation_type, std::vector<float> *features_min, std::vector<float> *features_max, std::vector<RunningStat> *running_avg_and_stddev);

            GenericData& operator=(GenericData const &gd);

            void normalise(int _normalisation_type, std::vector<float> *features_min, std::vector<float> *features_max, std::vector<RunningStat> *running_avg_and_stddev);

    };
}

#endif // GenericData_H

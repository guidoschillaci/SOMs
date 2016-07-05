/*
 * Generic Internal Model.
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
*/

#ifndef GenericInternalModel_H
#define GenericInternalModel_H

#include <vector>
#include <queue>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/multi_array.hpp>

#include "SOMs/GenericHebbianTable.h"
#include "Utils/NaoTH/Common.h"

namespace SOMs
{
    class GenericInternalModel
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////

            std::vector<GenericHebbianTable *> hebbianTables;

            // moving prediction error
            std::vector<float> moving_error;
            // derivative of moving prediction error
            std::vector<float> derivative_moving_error;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            GenericInternalModel() { }

            ////////////////////
            // VIRTUAL FUNCTIONS
            // none

            ////////////////////
            // OTHER METHODS
            void            connectHebbianTable(GenericHebbianTable* hebbianTable);

            GenericNeuron*  predict();

            // simple prediction takes the winner node in som1 and selects the best matched node in the som2
            // according to the function simplePredictInSom2
            float computePredictionError(std::vector<float> *output_pattern);

            void computeMovingError();
            void computeDerivativeOfMovingError();

            void loadMovingError(std::string filename);
            void saveMovingError(std::string filename);

    };

}

#endif // GenericInternalModel_H

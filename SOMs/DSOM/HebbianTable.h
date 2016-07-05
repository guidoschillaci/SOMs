/*
 * Hebbian Table.
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
*/

#ifndef HEBBIANTABLE_H
#define HEBBIANTABLE_H

#include <vector>
#include <queue>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/multi_array.hpp>
#include "SOMs/DSOM/Neuron.h"
#include "SOMs/DSOM/DSOM.h"
#include "SOMs/GenericHebbianTable.h"


namespace dsom
{
    class HebbianTable : public SOMs::GenericHebbianTable
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
//            dsom::DSOM *som1;
//            dsom::DSOM *som2;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            HebbianTable(SOMs::ParamHebbianTable *param, SOMs::GenericSOM *s1, SOMs::GenericSOM *s2)
                : SOMs::GenericHebbianTable(param, (dsom::DSOM*) s1, (dsom::DSOM*) s2) { }

            ////////////////////
            // METHODS OVERRIDED
//            float getActivationFromNodeInSOM1(int x, int y, std::vector<float> * input);
//            float getActivationFromNodeInSOM2(int x, int y, std::vector<float> * input);

            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS
            // none

            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS
            // none

    };
}

#endif // HEBBIANTABLE_H

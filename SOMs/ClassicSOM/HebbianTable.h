/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/

#ifndef ClassicSOM_HEBBIANTABLE_H
#define ClassicSOM_HEBBIANTABLE_H

#include <vector>
#include <queue>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/multi_array.hpp>
#include "SOMs/ClassicSOM/Neuron.h"
#include "SOMs/ClassicSOM/ClassicSOM.h"
#include "SOMs/GenericHebbianTable.h"


namespace classicSOM
{

    /**
     * @brief A specialisation of the Generic Hebbian Table
     *
     * A specialisation of the Generic Hebbian Table. In fact, everything as implemented in the ihnerithed class
     */
    class HebbianTable : public SOMs::GenericHebbianTable
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
            HebbianTable(SOMs::ParamHebbianTable *param, SOMs::GenericSOM *s1, SOMs::GenericSOM *s2)
                : SOMs::GenericHebbianTable(param, (classicSOM::ClassicSOM*) s1, (classicSOM::ClassicSOM*) s2) { }

            ////////////////////
            // METHODS OVERRIDED            


            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS
            // none

            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS
            // none

    };
}

#endif // PLSOM_HEBBIANTABLE_H


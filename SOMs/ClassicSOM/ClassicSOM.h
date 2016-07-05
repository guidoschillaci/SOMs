/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/

#ifndef ClassicSOM_H
#define ClassicSOM_H

#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <string>

#include "SOMs/GenericSOM.h"
#include "SOMs/GenericNeuron.h"
#include "SOMs/ClassicSOM/Neuron.h"

namespace classicSOM
{

    /**
     * @brief The ClassicSOM class implements the classical SOM algorithm (Kohonen map)
     *
     * This class implements the classical SOM algorithm (Kohonen map). After a number of iterations, the parameters learning_rate and sigma are
    * annealed exponentially.
     */
    class ClassicSOM : public SOMs::GenericSOM
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            std::vector<float> learning_rate;
            std::vector<float> sigma;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            ClassicSOM(std::string filename);
            ClassicSOM(SOMs::ParamSOM* _params);
            ClassicSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max);
            ClassicSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> * means, std::vector<float> *stddevs);

            ClassicSOM(const SOMs::GenericSOM& gsom);

            ////////////////////
            // METHODS OVERRIDED

            // Train step (only one input vector)
            float trainStep(std::vector<float> *input_pattern);
            float neighborhood_function(std::vector<float> *input_pattern, SOMs::GenericNeuron * winner, SOMs::GenericNeuron *current);

            void loadFromFile (std::string filename);
            void saveNodes(std::string filename);
            void savePredictionErrors(std::string filename);
            void loadPredictionErrors(std::string filename);


            ////////////////////
            // IMPLEMENTATION OF VIRTUAL FUNCTIONS IN THE GENERIC CLASS           
            void init();
            void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max);
            void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *means, std::vector<float> *stddev);
            ////////////////////
            // ADDITIONAL METHODS NOT PRESENT IN THE GENERIC CLASS            

    };

}

#endif // ClassicSOM_H

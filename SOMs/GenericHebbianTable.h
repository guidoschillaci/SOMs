/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#ifndef GenericHebbianTable_H
#define GenericHebbianTable_H

#include <vector>
#include <queue>
//#include <cmath>
//#include <cfloat>
#include <iostream>
//#include <fstream>
#include <string>

//#include <boost/multi_array.hpp>

#include "SOMs/GenericSOM.h"
#include "SOMs/ParamHebbianTable.h"
//#include "Utils/NaoTH/Common.h"

#define SIZE_LIST_OF_WINNERS    5

namespace SOMs
{
    /**
     * @brief The GenericHebbianTable class stores the Hebbian links connecting a pair of SOMs
     *
     * The GenericHebbianTable class stores the Hebbian links connecting a pair of SOMs.
     * A table has an input SOM and an output SOM, with links (axons) connecting each node
     * of the first to each node of the second.
     * A ParamHebbianTable object stores the configuration parameters of the table, i.e. the
     * prediction strategy or the learning strategy (i.e. update only the link between the winners
     * of the input and output maps, or all the link starting at the winner in the input map to all the
     * nodes in the output map?)
     *
     * The main training function is update().
     * For prediction, first clear the propagations to the output map, therefore propagate a desired
     * number of signals from the input map of this Hebbian Table (and/or from the input maps of other
     * Tables pointing to the same output map) to the output map.
     */
    class GenericHebbianTable
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            GenericSOM *inputSOM;
            GenericSOM *outputSOM;

            ParamHebbianTable   *param;

            // history of the learning rate
            std::vector<float> learning_rate;

            // Axons is a matrix containing all the hebbian links connecting the two SOMs
            // axons is a 4-D table (size: m*m*m*m): two dimensions for the som1 and two for the motor dsom
            std::vector< std::vector < std::vector < std::vector<float> > > > axons;


            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            //GenericHebbianTable();
            GenericHebbianTable(ParamHebbianTable *param, GenericSOM *inputSOM, GenericSOM *outputSOM);

            ////////////////////
            // VIRTUAL FUNCTIONS
            // none

            ////////////////////
            // OTHER METHODS            

            void connectSOMs(GenericSOM *inputSOM, GenericSOM *outputSOM); // used after loading

            // this is the function for updating the hebbian table using the input patterns
            void update(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor);

            // update the connection of the winner in one map to the winner in the other map
            void learnUsingWinners(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor);

            // update the connection of the winner and of its neighbours in one map to the winner and of its neighbours in the other map
            void learnUsingWinnersAndNeighbors(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor);

            // update all the connections
            void learnUsingAll(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor);

            // returns a list of the k best activated nodes in outputMap, selected from the connections from som1_winner in som1.
            // it returns also a list of normalised activations for the k neurons
            void getKActivatedOutputNodes(int k, GenericNeuron *input_winner,
                                             std::vector<GenericNeuron*> *activatedNodes,
                                             std::vector<float> *activations);

            // getBestActivatedOutputNode. Input is a pointer to a neuron
            GenericNeuron * getBestActivatedOutputNode(GenericNeuron *input_winner);

            // getBestActivatedOutputNode. Input is a pointer to an input_pattern
            GenericNeuron * getBestActivatedOutputNode(std::vector<float> *input_pattern);

            // getBestActivatedOutputNode. Input are the coordinates in the lattice of the node in som1
            GenericNeuron * getBestActivatedOutputNode(int node_x, int node_y);


            // propagate the activation of the input map to the output map
            // It does not still outputs a prediction, since the output map can input propagations from other maps
            float propagateInputToOutputMap(int node_x, int node_y);
            float propagateInputToOutputMap(std::vector<float> *input_pattern, bool normalise_input = false);
            // predict calls predict_simple() or predict_weightedAverage()
            GenericNeuron*  predict();
            // This outputs an unnormalised result
            GenericNeuron*  predictAndUnnormalise();
            // Get the best activated node in output map (activations are computed from propagated signals using propagateInputToOutputMap)
            GenericNeuron*  predict_simple();
            // Get the prediction as the weighted average of the nodes positions in output map (activations are computed from propagated signals using propagateInputToOutputMap)
            GenericNeuron*  predict_weightedAverage();

            // simple prediction takes the winner node in som1 and selects the best matched node in the som2
            // according to the function simplePredictInSom2
            float computePredictionError(std::vector<float> *output_pattern, std::vector<float> *input_pattern);

            float getPredictionError(std::vector<float> *output_pattern);

            // normalises all the weights
            void normaliseWeights();

            void loadTable(std::string filename);
            void saveTable(std::string filename);

            void printLinks();
            std::vector<float> getTable();
    };


}

#endif // GenericHebbianTable_H

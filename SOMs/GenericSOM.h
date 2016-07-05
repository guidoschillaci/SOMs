/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#ifndef GenericSOM_H
#define GenericSOM_H

#include <vector>
#include <queue>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "SOMs/GenericNeuron.h"
#include "SOMs/ParamSOM.h"


/**
 * Main library namespace.
 */
namespace SOMs
{


    /**
     * @brief The CompareWinners class is used in the function for getting the list of winners nearest to an input pattern
     */
    class CompareWinners {
    public:
         int operator() ( const std::pair<float, GenericNeuron *>& p1,
                          const std::pair<float, GenericNeuron *>& p2 ) {
             return p1.first > p2.first;
         }
    };

    /**
     * @brief The GenericSOM class implements a set of functions common to different SOM algorithms
     *
     * The GenericSOM class implements a set of functions common to different SOM algorithms.
     * In this version of the code, only a specialisation of the GenericSOM class exists:
     * the classicSOM, implementing the standard Kohonen SOM algorithm.
     */
    class GenericSOM
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            std::vector< std::vector<GenericNeuron*> > neurons;

            ParamSOM *params;

            float iteration;          // Number of training iterations so far

            // for feature scaling
            std::vector<float> features_min;
            std::vector<float> features_max;
            // for scaling the activation
//            float   distance_min;
//            float   distance_max;
            // the vector's size is equal to weight_dimensions
            std::vector<float> means;
            std::vector<float> stddevs;

            // number of nodes that have been visited at least once in updating the hebbian table
            std::vector<float> number_of_visited_nodes;

            // lookup table containing the squared distances in the lattice
            std::vector<std::vector<std::vector<std::vector<float> > > > lookup_squared_distances_in_lattice;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////


            ////////////////////
            // CONSTRUCTORS
            GenericSOM() { }
            GenericSOM(std::string filename);
            GenericSOM(ParamSOM *_params);
            GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max);
            GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *means, std::vector<float> *stddev);
            GenericSOM(const GenericSOM& gsom) { }


            ////////////////////
            // VIRTUAL FUNCTIONS

            // Train step (only one input vector)
            virtual void init() { std::cout<<"calling init from GenericSOM"<<std::endl; }
            virtual void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max) { std::cout<<"calling init from GenericSOM"<<std::endl;}
            virtual void init(std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *means, std::vector<float> *stddev) { std::cout<<"calling init from GenericSOM"<<std::endl;}

            void init_lookup_tables();

            std::vector<float> normalise_input(std::vector<float>*input);
            std::vector<float> normalise_input(std::vector<float>*input, int _normalisation_type, std::vector<float>* _features_min, std::vector<float>* _features_max);

            std::vector<float> unnormalise_input(std::vector<float>*input);
//            std::vector<float> unnormalise_input(std::vector<float>*input, int _normalisation_type, std::vector<float>* _features_min, std::vector<float>* _features_max, std::vector<RunningStat> *_running_avg_stddev);

            ////////////////////
            // OTHER METHODS
//            void populateRunningAvgStddev(std::vector<float> *means, std::vector<float> *stddev);
            float trainStep(std::vector<float> *input_pattern)
                { std::cerr<<"called trainStep from GenericSOM"<<std::endl; return -1;}
            float neighborhood_function(std::vector<float> *input_pattern, GenericNeuron * winner, GenericNeuron *current)
                { std::cerr<<"called neighborhood_function from GenericSOM"<<std::endl; return -1;}

            void loadFromFile (std::string filename);

            // steps to process before the training
            //void preTrainSteps(std::vector<float> *input_pattern);
            std::vector<float> preTrainSteps(std::vector<float> *input_pattern);

            // steps to process after the training
            void postTrainSteps(std::vector<float> *input_pattern);

            GenericNeuron * getWinner(std::vector<float> *input_pattern);
            bool getListOfWinners(std::vector<float> *input_pattern, std::priority_queue<std::pair<float, GenericNeuron *>, std::vector<std::pair<float, GenericNeuron *> >, SOMs::CompareWinners >
                                                                                    *list_of_winners);

            // returns the distance between the input pattern and the closest node to the input pattern
            // It needs to find the winner first
            float getDistanceToWinner(std::vector<float> *input_pattern);

            // Get neighbours of the node with position (x_pos,y_pos) in the lattice, considering a 2D SOM, with squared neighborhood
            std::vector<GenericNeuron*> * getNeighboursOf( int x_pos, int y_pos);

            float getSquaredDistance(std::vector<float> *pattern1, std::vector<float> *pattern2);

            float getDistance(std::vector<float> *pattern1, std::vector<float> *pattern2);

            float getSquaredDistanceBetweenInputAndNode(std::vector<float> *pattern1, int node_x, int node_y);

            float getDistanceBetweenInputAndNode(std::vector<float> *pattern1, int node_x, int node_y);

            // outputs a pointer to the node with highest input activation (the input activations are coming from propagated signals from other maps)
            GenericNeuron* getBestActivatedNode();
            void clearIncomingActivations();

            std::vector<float> getPropagatedActivations();
            std::vector<float> getActivationsFromInput(std::vector<float> *query, bool normalise_query);
            std::vector<float> getWeightsOfAllNodes();


            void printPropagatedActivations();
            void printNodes();

            void printNodesWithDistanceToQuery(std::vector<float> *query);

            void saveNodes(std::string filename);

            void savePredictionErrors(std::string filename);
            void loadPredictionErrors(std::string filename);

    };



}

#endif // GenericSOM_H

/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#ifndef ParamHebbianTable_H
#define ParamHebbianTable_H

// hebbian rule for updating the links
#define USE_POS_HEBBIAN_RULE                    1 // positive hebbian learning rule
#define USE_OJA_RULE                            2 // Oja's learning rule

// update strategy of the hebbian links
#define LEARN_USING_WINNERS                     0 // update only the link connecting the winner nodes in input and output map
#define LEARN_USING_WINNERS_AND_NEIGHBOURS      1 // update only the links between winner and its neighbours in input map and winner and its neighbors in output map
#define LEARN_USING_ALL                         2 // update all the links

// propagation type (from input map to output map)
// used in prediction processes
#define PROPAGATE_WINNER_TO_WINNER              0 // propagate only the activation of the winner in input map to the best connected node in output map
#define PROPAGATE_WINNER_TO_ALL                 1 // propagate the activation of the winner in input map to all the nodes in output map connected to it
#define PROPAGATE_K_WINNERS_TO_ALL              2 // propagate the activation of the closest k winners in input map to all the nodes in output map connected to it
#define PROPAGATE_ALL_TO_WINNERS                3 // propagate from each node in input map to their best connected nodes in output map
#define PROPAGATE_ALL_TO_ALL                    4 // propagate from each node in input map to each of the connected nodes in output map

// prediction type
#define PREDICTION_TYPE_SIMPLE                  0 // The predicted output is equal to the position of the winner node
#define PREDICTION_TYPE_WEIGHTED_AVERAGE        1 // The predicted output is equal to the weighted average of all the nodes' positions

namespace SOMs
{
     /**
     * @brief The ParamHebbianTable class stores configuration parameters
     * related to how Hebbian Tables are updated and used for predictions
     *
     * The ParamHebbianTable class stores configuration parameters
     * related to how Hebbian Tables are updated and used for predictions
     */
    class ParamHebbianTable
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            int hebbian_rule_type;
            float hebbian_learning_rate;
            int learning_strategy;
            bool normalise_hebbian_weights;
            int propagation_type;
            int prediction_type;
            int neighbours_size;
            int buffer_size;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            /**
             * @brief ParamHebbianTable initialises automatically the parameters for the Hebbian table
             */
            ParamHebbianTable()
            {
                hebbian_rule_type = USE_POS_HEBBIAN_RULE;
                hebbian_learning_rate = 0.01;//0.1;
                learning_strategy = LEARN_USING_WINNERS;
                normalise_hebbian_weights = true;

                propagation_type = PROPAGATE_K_WINNERS_TO_ALL;
                prediction_type = PREDICTION_TYPE_WEIGHTED_AVERAGE;

                neighbours_size=3;
                buffer_size = 100;// was300;
            }

            ////////////////////
            // OTHER METHODS
            /**
             * @brief printParams prints the current parameters
             */
            void printParams()
            {
                std::cout<<"--------------------------------"<<std::endl;
                std::cout<<"Parameters for Hebbian Learning."<<std::endl;
                std::cout<<"Hebbian rule type: ";
                if (hebbian_rule_type == USE_POS_HEBBIAN_RULE)
                    std::cout<<"positive hebb rule"<<std::endl;
                if (hebbian_rule_type == USE_OJA_RULE)
                    std::cout<<"oja's rule"<<std::endl;
                std::cout<<"Learning strategy: ";
                if (learning_strategy == LEARN_USING_WINNERS)
                    std::cout<<"using winners only"<<std::endl;
                if (learning_strategy == LEARN_USING_WINNERS_AND_NEIGHBOURS)
                    std::cout<<"using winners and neighbours"<<std::endl;
                if (learning_strategy == LEARN_USING_ALL)
                    std::cout<<"using all"<<std::endl;
                std::cout<<"Normalise hebbian weights: ";
                if (normalise_hebbian_weights)
                    std::cout<<"true"<<std::endl;
                else
                    std::cout<<"false"<<std::endl;
                std::cout<<"Propagation type: ";
                if (propagation_type == PROPAGATE_WINNER_TO_WINNER)
                    std::cout<<"winner-to-winner"<<std::endl;
                if (propagation_type == PROPAGATE_K_WINNERS_TO_ALL)
                    std::cout<<"k-winner-to-all"<<std::endl;
                if (propagation_type == PROPAGATE_WINNER_TO_ALL)
                    std::cout<<"winner-to-all"<<std::endl;
                if (propagation_type == PROPAGATE_ALL_TO_WINNERS)
                    std::cout<<"all-to-winners"<<std::endl;
                if (propagation_type == PROPAGATE_ALL_TO_ALL)
                    std::cout<<"all-to-all"<<std::endl;
                std::cout<<"Prediction type: ";
                if (prediction_type == PREDICTION_TYPE_SIMPLE)
                    std::cout<<"simple"<<std::endl;
                if (prediction_type == PREDICTION_TYPE_WEIGHTED_AVERAGE)
                    std::cout<<"weighted average"<<std::endl;
                std::cout<<"--------------------------------"<<std::endl;
            }
    };
}

#endif // ParamHebbianTable_H

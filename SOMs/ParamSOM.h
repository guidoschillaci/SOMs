/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/

#ifndef ParamSOM_H
#define ParamSOM_H

// normalisation of the input for the neuron activation function
#define NORMALISE_WEIGHTS_WITH_FEAT_SCALING       0 // scale the distance for calculating the activation between 0 and 1
#define NORMALISE_WEIGHTS_WITH_STANDARD_SCORE     1 // normalise each dimension using the corresponding mean and standard deviation

#define EUCLIDEAN_DISTANCE                        0
#define EUCLIDEAN_DISTANCE_WEIGHTED               1 // a factor is multiplied to each dimension. Used for MFCC, when giving more important to first coefficients

namespace SOMs
{

    /**
     * @brief The ParamSOM class stores configuration parameters for Self-Organising Maps
     *
     * The ParamSOM class stores configuration parameters for Self-Organising Maps
     */
    class ParamSOM
    {
        public:

            /////////////////////////////////////////
            // ATTRIBUTES
            /////////////////////////////////////////
            int som_size; // number of nodes = som_size*som_size
            int weights_size; // number of dimensions in each node

            int normalisation_type;
            int euclidean_distance_type;

            bool initialise_node_weights_with_zero_means_and_unit_stddev;

            // SOM_type specific attributes
            float classicSOM_initial_learning_rate;
            float classicSOM_initial_sigma;
            float classicSOM_constant_iterations;

            float DSOM_learning_rate;
            float DSOM_elasticity;

//            float PLSOM_beta;

            /////////////////////////////////////////
            // METHODS
            /////////////////////////////////////////

            ////////////////////
            // CONSTRUCTORS
            /**
             * @brief ParamSOM Initialises the parameters
             */
            ParamSOM() :
                initialise_node_weights_with_zero_means_and_unit_stddev(false),
                normalisation_type(NORMALISE_WEIGHTS_WITH_STANDARD_SCORE),
                euclidean_distance_type(EUCLIDEAN_DISTANCE),
                // SOM_type specific attributes
                classicSOM_constant_iterations (500),
                classicSOM_initial_sigma (0.7),
                classicSOM_initial_learning_rate(0.9),
                DSOM_learning_rate(0.2),
                DSOM_elasticity(1.5)
//                PLSOM_beta(1.4)
                { }
    };
}

#endif // ParamSOM_H

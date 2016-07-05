/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*
*
*/


#include "SOMs/GenericSOM.h"

namespace SOMs
{

    /**
     * @brief GenericSOM::GenericSOM The constructor of the Generic SOM calss
     * Deprecated. You should use  GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param filename the path of the model to load
     */
    GenericSOM::GenericSOM(std::string filename)
    {
//        distance_min = FLT_MAX;
//        distance_max = 0;
//        number_of_visited_nodes.clear();
//        loadFromFile(filename);
//        init_lookup_tables();
    }

    /**
     * @brief GenericSOM::GenericSOM Initialises the Generic SOM with ParamSOM
     * Deprecated. You should use  GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param _params The parameter object
     */
    GenericSOM::GenericSOM(ParamSOM *_params)
    {
//        params = _params;
//        distance_min = FLT_MAX;
//        distance_max = 0;
//        number_of_visited_nodes.clear();

//        // initialise weights randomly between -1 and 1
//        std::vector<float> ranges_min(params->weights_size, -1.0);
//        std::vector<float> ranges_max(params->weights_size, 1.0);

//        init(ranges_min, ranges_max);
//        init_lookup_tables();

//        // suppose the data has 0 mean and unitary stddev in each dimension
//        means.resize(params->weights_size, 0.0);
//        stddevs.resize(params->weights_size, 1.0);

    }

    /**
     * @brief GenericSOM::GenericSOM
     * Initialises the Generic SOMs
     * Deprecated. You should use  GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param _params The ParamSOM object
     * @param ranges_min a vector containing the minimum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param ranges_max a vector containing the maximum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     */
    GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max)
    {
//        params = _params;
//        distance_min = FLT_MAX;
//        distance_max = 0;
//        number_of_visited_nodes.clear();
//        init(ranges_min, ranges_max);
//        init_lookup_tables();

//        // suppose the data has 0 mean and unitary stddev in each dimension
//        means.resize(params->weights_size, 0.0);
//        stddevs.resize(params->weights_size, 1.0);
    }

    /**
     * @brief GenericSOM::GenericSOM
     * Initialises the Generic SOMs
     * Deprecated. You should use  GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param _params The ParamSOM object
     * @param ranges_min a vector containing the minimum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param ranges_max a vector containing the maximum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param _means a vector containing the means of the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param _stddev a vector containing the standard deviations of the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     */
    GenericSOM::GenericSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
    {
//        params = _params;
//        distance_min = FLT_MAX;
//        distance_max = 0;
//        number_of_visited_nodes.clear();
//        init(ranges_min, ranges_max);
//        init_lookup_tables();

//        means.resize(_means->size());
//        stddevs.resize(_stddev->size());

    }

    /**
     * @brief GenericSOM::init_lookup_tables
     * Lookup table for not computing every time the lattice distances between neurons
     */
    void GenericSOM::init_lookup_tables()
    {
        lookup_squared_distances_in_lattice.resize(params->som_size);
        for (int x1 = 0; x1 < params->som_size; x1++)
        {
            lookup_squared_distances_in_lattice[x1].resize(params->som_size);
            for (int y1 = 0; y1 < params->som_size; y1++)
            {
                lookup_squared_distances_in_lattice[x1][y1].resize(params->som_size);
                for (int x2 = 0; x2 < params->som_size; x2++)
                {
                    lookup_squared_distances_in_lattice[x1][y1][x2].resize(params->som_size);
                    for (int y2 = 0; y2 < params->som_size; y2++)
                    {
                        lookup_squared_distances_in_lattice[x1][y1][x2][y2]
                                = std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
                    }
                }
            }
        }
    }

    /**
     * @brief GenericSOM::normalise_input
     * Normalises the input with the method specified by paramsSOM->normalisation_type
     * @param input the input vector (the size of the vector should match paramSOM.weights_size)
     * @return the normalised output
     */
    std::vector<float> GenericSOM::normalise_input(std::vector<float> *input)
    {
        std::vector<float> normalised_input;
        if (input->size() != params->weights_size)
        {
            std::cerr<<"GenericSOM::normalise_input (input->size() != params->weights_size)"<<std::endl;
            return normalised_input;
        }


        if (params->normalisation_type == NORMALISE_WEIGHTS_WITH_FEAT_SCALING)
        {
            for (unsigned int i=0; i<input->size(); i++)
            {
                float den = features_max[i] - features_min[i];
                if ( den <= 0)
                    std::cerr<<"GenericNeuron::normalise_input - (features_max - features_min <= 0)"<<std::endl;
                normalised_input.push_back( (input->at(i) - features_min[i]) / den);
            }
        }
        else // if (params->normalisation_type == NORMALISE_ACTIV_WITH_RUNNING_AVG_STDDEV)
        {
            for (unsigned int i=0; i<input->size(); i++)
            {
                if (stddevs[i] == 0)
                {
                    normalised_input.push_back(input->at(i));
                }
                else
                    normalised_input.push_back((input->at(i) -means[i]) / stddevs[i]);
            }

        }
        return normalised_input;
    }

    /**
     * @brief GenericSOM::unnormalise_input
     * Un-normalises the input with the method specified by paramsSOM->normalisation_type
     * @param input the input vector (the size of the vector should match paramSOM.weights_size)
     * @return the normalised output
     */
    std::vector<float> GenericSOM::unnormalise_input(std::vector<float> *input)
    {
        std::vector<float> unnormalised_input;
        if (input->size() != params->weights_size)
        {
            std::cerr<<"GenericSOM::unnormalise_input (input->size() != params->weights_size)"<<std::endl;
            return unnormalised_input;
        }


        if (params->normalisation_type == NORMALISE_WEIGHTS_WITH_FEAT_SCALING)
        {
            for (unsigned int i=0; i<input->size(); i++)
            {
                float den = features_max[i] - features_min[i];
                if ( den <= 0)
                    std::cerr<<"GenericNeuron::unnormalise_input - (features_max - features_min <= 0)"<<std::endl;
                unnormalised_input.push_back( (input->at(i)* den) + features_min[i]);
            }
        }
        else // if (params->normalisation_type == NORMALISE_ACTIV_WITH_RUNNING_AVG_STDDEV)
        {
            for (unsigned int i=0; i<input->size(); i++)
            {
                if (stddevs[i] == 0)
                {
                    unnormalised_input.push_back(input->at(i));
                }
                else
                    unnormalised_input.push_back((input->at(i)*stddevs[i]) + means[i]) ;
            }

        }
        return unnormalised_input;
    }


//    /**
//     * @brief GenericSOM::populateRunningAvgStddev
//     * Populate the boost accumulator with 2000 samples sampled from a normal distribution with the means and stddev specified
//     * @param means the vector containing the mean of each dimension (the size of the vector should match paramSOM.weights_size)
//     * @param stddev the vector containing the stddev of each dimension (the size of the vector should match paramSOM.weights_size)
//     */
//    void GenericSOM::populateRunningAvgStddev(std::vector<float> *means, std::vector<float> *stddev)
//    {
//        if( (means->size()!=running_average_and_stddev.size()) || (stddev->size()!=running_average_and_stddev.size()))
//        {
//            std::cerr<<"GenericSOM constructor: ( (means->size()!=running_average_and_stddev.size()) || (stddev->size()!=running_average_and_stddev.size()))"<<std::endl;
//            return;
//        }
//        for (unsigned int i=0; i<running_average_and_stddev.size(); i++)
//        {
//            for (int j=0; j<2000; j++)
//            {
//                float value= Utils::normal(means->at(i),stddev->at(i));
//                running_average_and_stddev[i].Push(value);
//            }
//        }
//    }

    /**
     * @brief GenericSOM::loadFromFile
     * Load the model from a file
     * @param filename the path to the file (include the name)
     */
    void GenericSOM::loadFromFile (std::string filename)
    {
//        std::cout<<"Loading SOM from "<<filename.c_str()<<std::endl;
//        std::ifstream inputfile;
//        inputfile.open(filename.c_str(), std::ifstream::in);

//        std::string line;
//        // header (som_size weight_dimensions)
//        getline (inputfile, line);
//        getline (inputfile, line);

//        std::istringstream iss_header(line);
//        char c;
//        iss_header>> c; // skip the # character
//        iss_header>>params->som_size;
//        iss_header>>params->weights_size;

//        // skip second header
//        getline (inputfile, line);

//        init();

//        int x, y;
//        std::vector<float> w;
//        w.resize(params->weights_size);

//        while (getline (inputfile, line))
//        {
//            std::istringstream iss(line);
//            iss>>x;
//            iss>>y;
//            for (int i=0; i<params->weights_size; i++)
//                iss>>neurons[x][y]->weights[i];
//            iss>>neurons[x][y]->number_of_visits;
//            float value;
//        }

//        std::cout<<"SOM loaded."<<std::endl;

        std::cerr<< "Calling GenericSOM::loadFromFile. You should use the load function from a specialised SOM class"<<std::endl;
    }

    /**
     * @brief GenericSOM::preTrainSteps
     * steps to process before the training, such as normalising the input pattern
     * @param input_pattern the input pattern
     * @return returns the normalised input pattern
     */
    std::vector<float> GenericSOM::preTrainSteps(std::vector<float> *input_pattern)
    {
        return normalise_input(input_pattern);
    }

    /**
     * @brief GenericSOM::postTrainSteps
     * steps to process after the training
     * @param input_pattern
     */
    void GenericSOM::postTrainSteps(std::vector<float> *input_pattern)
    {
        iteration++;

    }

    /**
     * @brief GenericSOM::getWinner
     * Returns a pointer to the closest node to the input pattern
     * @param input_pattern
     * @return a pointer to the closest node to the input pattern
     */
    GenericNeuron * GenericSOM::getWinner(std::vector<float> *input_pattern)
    {
        GenericNeuron *winner = neurons[0][0];

        float min = FLT_MAX;
        for (int i = 0; i < params->som_size; i++)
        {
            for (int j = 0; j < params->som_size; j++)
            {
                float d = getSquaredDistance(input_pattern, &neurons[i][j]->weights);
                if (d < min)
                {
                    min = d;
                    winner = (neurons[i][j]);
                }
            }
        }

        return winner;
    }

    /**
     * @brief GenericSOM::getListOfWinners
     * Order the nodes according to their distance to the input pattern. Put the output into list_of_winners
     * @param input_pattern
     * @param list_of_winners a std::priority_queue object containing an ordered list of the nodes, from the closest to the input pattern to the most far
     * @return
     */
    bool GenericSOM::getListOfWinners(std::vector<float> *input_pattern, std::priority_queue<std::pair<float, GenericNeuron *>, std::vector<std::pair<float, GenericNeuron *> >, CompareWinners > *list_of_winners)
    {
        for (int i = 0; i < params->som_size; i++)
        {
            for (int j = 0; j < params->som_size; j++)
            {
                float d = getSquaredDistance(input_pattern, &neurons[i][j]->weights);
                list_of_winners->push(std::pair<float, GenericNeuron*>(d, neurons[i][j]));
            }
        }
    }


    /**
     * @brief GenericSOM::getDistanceToWinner
     * returns the distance between the input pattern and the closest node to the input pattern
     * It needs to find the winner first
     * @param input_pattern
     * @return the distance to the winner node
     */
    float GenericSOM::getDistanceToWinner(std::vector<float> *input_pattern)
    {
        float min = FLT_MAX;
        for (int i = 0; i < params->som_size; i++)
        {
            for (int j = 0; j < params->som_size; j++)
            {
                float d = getSquaredDistance(input_pattern, &neurons[i][j]->weights);
                if (d < min)
                {
                    min = d;
                }
            }
        }
        return sqrt(min);
    }

    /**
     * @brief GenericSOM::getNeighboursOf
     * Get neighbours of the node with position (x_pos,y_pos) in the lattice
     * @param x_pos
     * @param y_pos
     * @return a vector containing the neighbors of the node in the lattice
     */
    std::vector<GenericNeuron*> * GenericSOM::getNeighboursOf( int x_pos, int y_pos)
    {
        std::cout<<"Calling getNeighboursOf from genericsom"<<std::endl;
        std::vector<GenericNeuron*> *neighbours;

        for ( int x = (x_pos - 1); x <= (x_pos + 1); ++x)
        {
            for ( int y = (y_pos -1); y <= (y_pos+1); ++y)
            {
                if (!( (x<0) || (y<0) || (x>=params->som_size) || (y>=params->som_size) ) )
                {
                    neighbours->push_back(neurons[x][y]);
                }
            }
        }
        return neighbours;
    }

    /**
     * @brief GenericSOM::getSquaredDistance
     * get the squared distance between patter1 and pattern2
     * @param pattern1
     * @param pattern2
     * @return
     */
    float GenericSOM::getSquaredDistance(std::vector<float> *pattern1, std::vector<float> *pattern2)
    {
        // this function does not depend on the neuron's weight,
        // thus call it for the first neuron
        return neurons[0][0]->getSquaredDistance(pattern1, pattern2, &features_min, &features_max);
    }

    /**
     * @brief GenericSOM::getDistance
     * get the distance between patter1 and pattern2
     * @param pattern1
     * @param pattern2
     * @return
     */
    float GenericSOM::getDistance(std::vector<float> *pattern1, std::vector<float> *pattern2)
    {
        float value = getSquaredDistance(pattern1, pattern2);
        return sqrt(value);
    }

    /**
     * @brief GenericSOM::getSquaredDistanceBetweenInputAndNode
     * get the squared distance between an input pattern and the weights of the node with lattice position node_x and node_y
     * @param pattern1
     * @param node_x
     * @param node_y
     * @return
     */
    float GenericSOM::getSquaredDistanceBetweenInputAndNode(std::vector<float> *pattern1, int node_x, int node_y)
    {
        return getSquaredDistance(pattern1, &neurons[node_x][node_y]->weights);
    }

    /**
     * @brief GenericSOM::getDistanceBetweenInputAndNode
     * get the distance between an input pattern and the weights of the node with lattice position node_x and node_y
     * @param pattern1
     * @param node_x
     * @param node_y
     * @return
     */
    float GenericSOM::getDistanceBetweenInputAndNode(std::vector<float> *pattern1, int node_x, int node_y)
    {
        return sqrt(getSquaredDistanceBetweenInputAndNode(pattern1, node_x, node_y));
    }

    /**
     * @brief GenericSOM::getBestActivatedNode
     * return a GenericNeuron pointer to the node with highest input activation (the input activations are coming from propagated signals from other maps)
     * @return
     */
    GenericNeuron* GenericSOM::getBestActivatedNode()
    {
        GenericNeuron *bestActivatedNode = neurons[0][0];
        float max = 0;

        for (unsigned int x = 0; x < params->som_size; x++)
        {
            for (unsigned int y = 0; y < params->som_size; y++)
            {
                if (neurons[x][y]->propagated_activation > max)
                {
                    max = neurons[x][y]->propagated_activation;
                    bestActivatedNode = neurons[x][y];
                }
            }
        }
        if (max == 0)
            std::cerr<<"GenericSOM::getBestActivatedNode() : all the nodes have incoming activations equal to 0. Did you propagate any signal to this map?"<<std::endl;
        return bestActivatedNode;
    }

    /**
     * @brief GenericSOM::clearIncomingActivations
     * clear the activations propagated from other maps to the nodes of this one
     * Run this command before each propagation, if you do not want to accumulate multiple propagations
     */
    void GenericSOM::clearIncomingActivations()
    {
        for (unsigned int x = 0; x < params->som_size; x++)
        {
            for (unsigned int y = 0; y < params->som_size; y++)
            {
                neurons[x][y]->propagated_activation = 0;
            }
        }
    }

    /**
     * @brief GenericSOM::printNodes
     * prints the weights of all the nodes in this map
     */
    void GenericSOM::printNodes()
    {
        std::cout<<"x_node y_node weights"<<std::endl;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                std::cout<<i<<" "<<j<<" ";
                for (int w=0; w<neurons[i][j]->weights.size(); w++)
                {
                    std::cout<< neurons[i][j]->weights[w]<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }

    /**
     * @brief GenericSOM::getPropagatedActivations
     * Returns a pointer to a vector containing the activation received for each node from other SOMs
     * The lattice is reshaped to a one-dimensional vector: row1, row2, row3, etc.
     * @return
     */
    std::vector<float> GenericSOM::getPropagatedActivations()
    {
        std::vector<float> output;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                output.push_back(neurons[i][j]->propagated_activation);
            }
        }
        return output;
    }

    /**
     * @brief GenericSOM::getActivationsFromInput
     * Returns a pointer to a vector containing the activation of each node of the map given an input
     * The lattice is reshaped to a one-dimensional vector: row1, row2, row3, etc.
     * @param query
     * @return
     */
    std::vector<float> GenericSOM::getActivationsFromInput(std::vector<float> *input_pattern, bool normalise_query)
    {
        std::vector<float> output;
        if (input_pattern->size() != params->weights_size)
        {
            std::cerr<<"GenericSOM::getActivationsFromInput: (input_pattern->size() != params->weights_size)"<<std::endl;
            return output;
        }

        std::vector<float> normalised_input;
        std::vector<float> *input;
        if (normalise_query)
        {
            normalised_input = normalise_input(input_pattern);
            input = &normalised_input;
        }
        else
            input = input_pattern;

        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                output.push_back(getSquaredDistance(input, &neurons[i][j]->weights) );
            }
        }
        return output;
    }

    /**
     * @brief GenericSOM::getWeightsOfAllNodes
     * This returns the weights of all nodes
     * The lattice is reshaped to a one-dimensional vector: row1, row2, row3, etc.
     * Each rows contains the weights of each nodes (size changes according to the weights size)
     * @return
     */
    std::vector<float> GenericSOM::getWeightsOfAllNodes()
    {
        std::vector<float> output;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                for (int w=0; w<params->weights_size; w++)
                    output.push_back( neurons[i][j]->weights.at(w) );
            }
        }
        return output;
    }

    /**
     * @brief GenericSOM::printPropagatedActivations
     * prints the activations of each node of the map
     */
    void GenericSOM::printPropagatedActivations()
    {
        std::cout<<"Propagated activations"<<std::endl;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                std::cout<< neurons[i][j]->propagated_activation<<" ";
            }
            std::cout<<std::endl;
        }
    }


    /**
     * @brief GenericSOM::printNodesWithDistanceToQuery
     * Print the weights of each node and their distance to a query input pattern (for debugging)
     * @param query
     */
    void GenericSOM::printNodesWithDistanceToQuery(std::vector<float> *query)
    {
        std::cout<<"x_node y_node weights distance_to_query"<<std::endl;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                std::cout<<i<<" "<<j<<" ";
                for (int w=0; w<neurons[i][j]->weights.size(); w++)
                {
                    std::cout<< neurons[i][j]->weights[w]<<" ";
                }
                std::cout<<getDistance(&neurons[i][j]->weights, query)<<std::endl;
            }
        }
    }


}

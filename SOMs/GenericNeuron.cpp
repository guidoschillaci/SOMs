/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#include "SOMs/GenericNeuron.h"

namespace SOMs
{

    /**
     * @brief GenericNeuron::operator =
     * Returns a Neuron with the same weights of n
     * @param n
     * @return
     */
    GenericNeuron& GenericNeuron::operator=(GenericNeuron const &n)
    {
        propagated_activation=0;
        X = n.X;
        Y = n.Y;
        number_of_visits=0;
//        number_of_victories_in_predictions = 0;
        weights.resize(n.weights.size());
        for (int k = 0; k < n.weights.size(); k++)
            weights[k] = n.weights[k];
    }

    /**
     * @brief GenericNeuron::init
     * Initialise the neuron with random weights (sampled from uniform distribution [0,1])
     * @param x the x coordinate in the lattice
     * @param y the y coordinate in the lattice
     * @param _params the params file
     */
    void GenericNeuron::init(int x, int y,  ParamSOM* _params)
    {
        params = _params;
        propagated_activation=0;
        // set GenericNeuron position in the DSOM grid
        X = x;
        Y = y;
        number_of_visits=0;
//        number_of_victories_in_predictions = 0;
        // initialize weights
        weights.resize(params->weights_size);
        for (int k = 0; k < params->weights_size; k++)
            weights[k] = ((float) rand() / (RAND_MAX)); // between 0 and 1

    }

    /**
     * @brief GenericNeuron::init
     * Initialise the neuron with random weights (each dimension i sampled from uniform distribution [min(i),max(i)])
     * @param x the x coordinate in the lattice
     * @param y the y coordinate in the lattice
     * @param _params the params file
     * @param min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     */
    void GenericNeuron::init(int x, int y, ParamSOM* _params, std::vector<float> *min, std::vector<float> *max)
    {        
        params = _params;
        if (min->size()!=params->weights_size || max->size()!=params->weights_size )
        {
            std::cerr<<"void GenericNeuron::init(int x, int y, ParamSOM* _params, std::vector<float> *min, std::vector<float> *max): (min->size()!=params->weights_size || max->size()!=params->weights_size )"<<std::endl;
            return;
        }
        propagated_activation=0;
        // set neuron positions in the SOM grid
        X = x;
        Y = y;
        number_of_visits=0;
//        number_of_victories_in_predictions = 0;
        // initialize weights
        weights.resize(params->weights_size);
        for (int k = 0; k < params->weights_size; k++)
        {
            weights[k] = (((float) rand() / (RAND_MAX)) * (max->at(k)-min->at(k))) + min->at(k); // between min and max
        }
    }

    // initialise the neuron's weight sampling from a normal distribution with desired means and stddev
    /**
     * @brief GenericNeuron::init
     * Initialise the neuron with random weights (each dimension i sampled from a normal distribution with desired means and stddev)
     * @param x the x coordinate in the lattice
     * @param y the y coordinate in the lattice
     * @param _params the params file
     * @param min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @param means  a vector of containing the means for each dimension (size should match params->weights_size )
     * @param stddev  a vector of containing the standard deviations for each dimension (size should match params->weights_size )
     */
    void GenericNeuron::init(int x, int y, ParamSOM* _params, std::vector<float> *min, std::vector<float> *max, std::vector<float> *means, std::vector<float> *stddev)
    {
        params = _params;
        propagated_activation=0;
        // set neuron positions in the SOM grid
        X = x;
        Y = y;
        number_of_visits=0;
//        number_of_victories_in_predictions = 0;
        // initialize weights
        weights.resize(params->weights_size);
        for (int k = 0; k < params->weights_size; k++)
        {
            weights[k] = Utils::normal(means->at(k),stddev->at(k));
        }
    }

    /**
     * @brief GenericNeuron::getDistance
     * get the distance between patt1 and patt2. It requires additional inputs that are not used, since the input pattern should be already normalised
     * @param patt1
     * @param patt2
     * @param features_min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param features_max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @return the distance
     */
    float GenericNeuron::getDistance(std::vector<float> *patt1, std::vector<float> *patt2, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        return std::sqrt( getSquaredDistance(patt1, patt2, features_min, features_max) );
    }

    /**
     * @brief GenericNeuron::getDistance
     * get the distance between the weights of the current to and an input pattern. It requires additional inputs that are not used, since the input pattern should be already normalised
     * @param input_pattern
     * @param features_min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param features_max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @return the distance
     */
    float GenericNeuron::getDistance(std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        return std::sqrt( getSquaredDistance(&weights, input_pattern, features_min, features_max) );
    }


    /**
     * @brief GenericNeuron::getSquaredDistance
     * get the squared distance between patt1 and patt2. It requires additional inputs that are not used, since the input pattern should be already normalised
     * @param patt1
     * @param patt2
     * @param features_min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param features_max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @return the distance
     */
    float GenericNeuron::getSquaredDistance(std::vector<float> *patt1, std::vector<float> *patt2, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        if (patt1->size()!=patt2->size())
            std::cerr<<"GenericNeuron: Error in getSquaredDistance. (patt1->size() "<<patt1->size()<<" != patt2->size() "<<patt2->size()<<")"<<std::endl;
        if (patt1->size()==0)
            std::cerr<<"GenericNeuron: Error in getSquaredDistance. patt1 size is 0"<<std::endl;

        float value = 0;
        float alpha;
        if (params->euclidean_distance_type == EUCLIDEAN_DISTANCE)
            alpha = 1;
        for (unsigned int i = 0; i < patt1->size(); i++)
        {
            if (params->euclidean_distance_type == EUCLIDEAN_DISTANCE_WEIGHTED)
                alpha = pow(0.95, i);
            value += alpha * std::pow((patt1->at(i) - patt2->at(i)), 2);
        }
        return value;

    }

    /**
     * @brief GenericNeuron::getSquaredDistance
     * get the squared distance between the weights of the current to and an input pattern. It requires additional inputs that are not used, since the input pattern should be already normalised
     * @param input_pattern
     * @param features_min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param features_max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @return the distance
     */
    float GenericNeuron::getSquaredDistance(std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        return getSquaredDistance(&weights, input_pattern, features_min, features_max);
    }

    /**
     * @brief GenericNeuron::getActivation
     * Get the activation of the current node, which is function of its distance to an input pattern.  It requires additional inputs that are not used, since the input pattern should be already normalised
     * @param input_pattern
     * @param features_min a vector of containing the minimum values for each dimension (size should match params->weights_size )
     * @param features_max a vector of containing the maximum values for each dimension (size should match params->weights_size )
     * @param running_avg_and_stddev a vector of containing the statistics for each dimension (size should match params->weights_size )
     * @return the activation
     */
    float GenericNeuron::getActivation (std::vector<float> *input_pattern, std::vector<float> *features_min, std::vector<float> *features_max)
    {
        float distance = getDistance(input_pattern, features_min, features_max);
        return  1/(1+2*std::tanh(std::abs(distance)));
    }


    /**
     * @brief GenericNeuron::weightsEqualTo
     * Checks if the weights of the this node are equals to an input pattern
     * @param input
     * @return
     */
    bool GenericNeuron::weightsEqualTo(std::vector<float>* input)
    {
        if (input->size()!=weights.size())
            return false;
        for (unsigned int i=0; i< input->size(); i++)
            if (input->at(i)!=weights[i])
                return false;
        return true;
    }

    /**
     * @brief GenericNeuron::printWeights
     * Prints the position of the node in the lattice and its weights
     */
    void GenericNeuron::printWeights()
    {
        std::cout<<"Node coords (in lattice) "<<X<<" "<<Y<<" weights ";
        for (unsigned int i=0; i<weights.size(); i++)
        {
            std::cout<<weights[i]<<" ";
        }
        std::cout<<std::endl;
    }
}

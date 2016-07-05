/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#include "ClassicSOM.h"

namespace classicSOM
{

    /**
     * @brief ClassicSOM::ClassicSOM
     * The constructor of the ClassicSOM class
     * Deprecated. You should use  ClassicSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param filename the path of the model to load
     */
    ClassicSOM::ClassicSOM(std::string filename)
    {
        number_of_visited_nodes.clear();
        loadFromFile(filename);
        init_lookup_tables();
    }

    /**
     * @brief ClassicSOM::ClassicSOM
     * Initialises the ClassicSOM with ParamSOM
     * The weights of each node are randomly initialised between -1 and 1
     * Means and stddev of each dimension are set to 0 and 1, respectively
     * @param _params The parameter object
     */
    ClassicSOM::ClassicSOM(SOMs::ParamSOM* _params)
    {
        params = _params;
        number_of_visited_nodes.clear();

        // initialise weights randomly between -1 and 1
        std::vector<float> ranges_min(params->weights_size, -1.0);
        std::vector<float> ranges_max(params->weights_size, 1.0);

        init(&ranges_min, &ranges_max);
        init_lookup_tables();

        // suppose the data has 0 mean and unitary stddev in each dimension
        means.resize(params->weights_size, 0.0);
        stddevs.resize(params->weights_size, 1.0);

    }

    /**
     * @brief ClassicSOM::ClassicSOM
     * Initialises the ClassicSOM
     * initialises the weights of the nodes within the ranges specified
     * Means and stddev of each dimension are set to 0 and 1, respectively
     * @param _params The ParamSOM object
     * @param ranges_min a vector containing the minimum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param ranges_max a vector containing the maximum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     */
    ClassicSOM::ClassicSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max)
    {
        params = _params;
        number_of_visited_nodes.clear();
        init(ranges_min, ranges_max);
        init_lookup_tables();

        // suppose the data has 0 mean and unitary stddev in each dimension
        means.resize(params->weights_size, 0.0);
        stddevs.resize(params->weights_size, 1.0);
    }

    /**
     * @brief ClassicSOM::ClassicSOM
     * Initialises the ClassicSOM
     * Deprecated. You should use  ClassicSOM::ClassicSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param _params The ParamSOM object
     * @param ranges_min a vector containing the minimum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param ranges_max a vector containing the maximum values for the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param _means a vector containing the means of the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     * @param _stddev a vector containing the standard deviations of the weights of the nodes (the size of the vector should match paramSOM.weights_size)
     */
    ClassicSOM::ClassicSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddevs)
    {
        params = _params;
        number_of_visited_nodes.clear();
        init(ranges_min, ranges_max, _means, _stddevs);
        init_lookup_tables();        
    }

    /**
     * @brief ClassicSOM::ClassicSOM
     * copy constructor
     * Deprecated. You should use  ClassicSOM::ClassicSOM(ParamSOM *_params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
     * @param gsom
     */
    ClassicSOM::ClassicSOM(const SOMs::GenericSOM& gsom)
    {
        params = gsom.params;
        number_of_visited_nodes.clear();
        features_min.resize(params->weights_size);
        features_max.resize(params->weights_size);
        means.resize(params->weights_size,0.0);
        stddevs.resize(params->weights_size,1.0);

        for (unsigned int i =0; i<params->weights_size; i++)
        {
            features_min[i]= gsom.features_min[i];
            features_max[i]= gsom.features_max[i];
        }
        init(&features_min, &features_max);
        for (int i = 0; i < params->som_size; i++)
        {
            for (int j = 0; j < params->som_size; j++)
            {
                neurons[i][j] = new classicSOM::Neuron(*gsom.neurons[i][j]);
            }
        }
        init_lookup_tables();
    }

    /**
     * @brief ClassicSOM::init
     * Initialise the parameters of the SOM and its weights
     */
    void ClassicSOM::init()
    {
        iteration = 0;
        features_min.resize(params->weights_size);
        features_max.resize(params->weights_size);        
        means.resize(params->weights_size);
        stddevs.resize(params->weights_size);

        // weights of neurons are initialised between 0 and 1
        for (unsigned int i =0; i<params->weights_size; i++)
        {
            features_min[i]= 0;
            features_max[i]= 1;
        }

        // compute the max error as the maximum distance possible in the feature space
        float max_error = 0;
        for (unsigned int i = 0; i< params->weights_size; i++)
        {
            max_error += pow (features_max[i] - features_min[i], 2);
        }
        max_error = sqrt (max_error);

        neurons.resize(params->som_size);
        for (int i = 0; i < params->som_size; i++)
        {
            neurons[i].resize(params->som_size);
            for (int j = 0; j < params->som_size; j++)
            {
                neurons[i][j] = new classicSOM::Neuron();
                ((classicSOM::Neuron*)neurons[i][j])->init(i,j, params);
            }
        }
    }

    /**
     * @brief ClassicSOM::init
     * Initialise the parameters of the SOM and its weights
     * @param ranges_min
     * @param ranges_max
     */
    void ClassicSOM::init(std::vector<float> *ranges_min, std::vector<float> *ranges_max)
    {
        if (ranges_min->size()!=params->weights_size || ranges_max->size()!= params->weights_size)
            std::cerr<<"ranges_min->size()!=weight_dimensions || ranges_max->size()!=weight_dimensions"<<std::endl;

        iteration = 0;

        features_min.resize(params->weights_size);
        features_max.resize(params->weights_size);
        means.resize(params->weights_size);
        stddevs.resize(params->weights_size);

        for (unsigned int i =0; i<params->weights_size; i++)
        {
            features_min[i]= ranges_min->at(i);
            features_max[i]= ranges_max->at(i);
        }

        // compute the max error as the maximum distance possible in the feature space
        float max_error = 0;
        for (unsigned int i = 0; i< params->weights_size; i++)
        {
            max_error += pow (features_max[i] - features_min[i], 2);
        }
        max_error = sqrt (max_error);

        neurons.resize(params->som_size);
        for (int i = 0; i < params->som_size; i++)
        {
            neurons[i].resize(params->som_size);
            for (int j = 0; j < params->som_size; j++)
            {
                neurons[i][j] = new classicSOM::Neuron();
                ((classicSOM::Neuron*)neurons[i][j])->init(i,j, params, ranges_min, ranges_max);
            }
        }
    }

    /**
     * @brief ClassicSOM::init
     * Initialise the parameters of the SOM and its weights
     * @param ranges_min
     * @param ranges_max
     * @param _means
     * @param _stddev
     */
    void ClassicSOM::init(std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
    {
        if (ranges_min->size()!=params->weights_size || ranges_max->size()!=params->weights_size  || _means->size()!=params->weights_size || _stddev->size()!=params->weights_size )
            std::cerr<<"ranges_min->size()!=weight_dimensions || ranges_max->size()!=weight_dimensions || means->size()!=weight_dimensions || stddev->size()!=weight_dimensions"<<std::endl;

        iteration = 0;

        means.resize(params->weights_size);
        stddevs.resize(params->weights_size);
        for (unsigned int i=0; i< _means->size(); i++)
        {
            means[i] = _means->at(i);
            stddevs[i] = _stddev->at(i);
        }

        features_min.resize(params->weights_size );
        features_max.resize(params->weights_size );

        for (unsigned int i =0; i<params->weights_size ; i++)
        {
            features_min[i]= ranges_min->at(i);
            features_max[i]= ranges_max->at(i);
        }

        // compute the max error as the maximum distance possible in the feature space
        float max_error = 0;
        for (unsigned int i = 0; i< params->weights_size ; i++)
        {
            max_error += pow (features_max[i] - features_min[i], 2);
        }
        max_error = sqrt (max_error);

        std::vector<float> zero_means;
        zero_means.resize(params->weights_size , 0.0);
        std::vector<float> unit_stddev;
        unit_stddev.resize(params->weights_size , 1.0);

        neurons.resize(params->som_size);
        for (int i = 0; i < params->som_size; i++)
        {
            neurons[i].resize(params->som_size);
            for (int j = 0; j < params->som_size; j++)
            {
                neurons[i][j] = new classicSOM::Neuron();
                if (params->initialise_node_weights_with_zero_means_and_unit_stddev)
                    ((classicSOM::Neuron*)neurons[i][j])->init(i,j,params, ranges_min, ranges_max, &zero_means, &unit_stddev);
                else
                    ((classicSOM::Neuron*)neurons[i][j])->init(i,j,params, ranges_min, ranges_max, &means, &stddevs);
            }
        }
    }

    /**
     * @brief ClassicSOM::loadFromFile
     * load the SOM from a file
     * @param filename
     */
    void ClassicSOM::loadFromFile (std::string filename)
    {
        std::cout<<"Loading ClassicSOM from "<<filename.c_str()<<std::endl;
        std::ifstream inputfile;
        inputfile.open(filename.c_str(), std::ifstream::in);
        if (!inputfile.is_open())
        {
            std::cerr<<"Could not open "<<filename.c_str()<<"\n";
            return;
        }

        std::string line;
        // header (som_size weight_dimensions learning_rate elasticity)        
        getline (inputfile, line);

        getline (inputfile, line);
        std::istringstream iss_header(line);
        char c;
        iss_header>> c; // skip the # character
        iss_header>>params->som_size;
        iss_header>>params->weights_size;
//        std::cout<<"params->som_size "<<params->som_size<<" params->weights_size "<<params->weights_size<<std::endl;
        iss_header>>params->classicSOM_initial_learning_rate;
        iss_header>>params->classicSOM_initial_sigma;
        iss_header>>params->classicSOM_constant_iterations;
        float nr_iterations;
        iss_header>>nr_iterations;

        init();
        iteration = nr_iterations;

        float val;
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        for (unsigned int i = 0; i<features_min.size(); i++)
        {
            iss_header>>features_min[i];
        }
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        for (unsigned int i = 0; i<features_max.size(); i++)
        {
            iss_header>>features_max[i];
        }

        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
//        iss_header>>distance_min;
//        iss_header>>distance_max;

        std::vector<float> temp_mean;
        temp_mean.resize(means.size());
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        std::cout<<"Means: ";
        for (unsigned int i = 0; i<means.size(); i++)
        {
            iss_header>>temp_mean[i];
            means[i] = temp_mean[i];
            std::cout<<temp_mean[i]<<", ";
        }
        std::cout<<std::endl;

        std::vector<float> temp_stddev;
        temp_stddev.resize(stddevs.size());
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        std::cout<<"Stddev: ";
        for (unsigned int i = 0; i<stddevs.size(); i++)
        {
            iss_header>>temp_stddev[i];
            stddevs[i] = temp_stddev[i];
            std::cout<<temp_stddev[i]<<", ";
        }
        std::cout<<std::endl;

        // skip second header
        getline (inputfile, line);

        int x, y;
        while (getline (inputfile, line))
        {
            std::istringstream sts(line);
            sts>>x;
            sts>>y;
            for (int i=0; i<params->weights_size; i++)
            {
                sts>>neurons[x][y]->weights[i];
            }
            sts>>neurons[x][y]->number_of_visits;
            float value;
        }

        std::cout<<"ClassicSOM loaded."<<std::endl;
    }

    /**
     * @brief ClassicSOM::saveNodes
     * Save the map into a file
     * @param filename
     */
    void ClassicSOM::saveNodes(std::string filename)
    {
        std::ofstream myfile;
        myfile.open (filename.c_str());
        myfile<<"# som_size weight_dimensions init_learning_rate init_sigma constant_iterations nr_interations_already_done"<<std::endl;
        myfile<<"# "<<params->som_size<<" "<< params->weights_size<<" "<< params->classicSOM_initial_learning_rate<<" "<<params->classicSOM_initial_sigma<<" " <<params->classicSOM_constant_iterations<<" "<< iteration<<std::endl;
        myfile<<"# features_min "<<std::endl<<"# ";
        for (unsigned int i = 0; i<features_min.size();i++)
        {
            myfile<<features_min[i]<<" ";
        }
        myfile<<std::endl;
        myfile<<"# features_max "<<std::endl<<"# ";
        for (unsigned int i = 0; i<features_max.size();i++)
        {
            myfile<<features_max[i]<<" ";
        }
        myfile<<std::endl;
        myfile<<"# distance_min distance_max"<<std::endl<<"# ";
        //myfile<<distance_min<<" "<<distance_max<<std::endl;
        myfile<<" "<<std::endl;

        myfile<<"# means "<<std::endl<<"# ";
        for (unsigned int i = 0; i<means.size();i++)
        {
            myfile<<means[i]<<" ";
        }
        myfile<<std::endl;
        myfile<<"# std_dev "<<std::endl<<"# ";
        for (unsigned int i = 0; i<stddevs.size();i++)
        {
            myfile<<stddevs[i]<<" ";
        }
        myfile<<std::endl;
        myfile<<"# x y [ weights ] number_of_visits"<<std::endl;
        for (int i=0; i<params->som_size; i++)
        {
            for (int j=0; j<params->som_size; j++)
            {
                myfile<<i<<" "<<j<<" ";
                for (int w=0; w<neurons[i][j]->weights.size(); w++)
                {
                    myfile<< neurons[i][j]->weights[w]<<" ";
                }

                myfile<<neurons[i][j]->number_of_visits<<std::endl;
            }
        }
        myfile.close();
    }

    /**
     * @brief ClassicSOM::trainStep
     * This implements the update rule of the classic SOM algorithm (Kohonen)
     * @param non_normalised_input_pattern
     * @return an error (the total distance travelled by all the nodes)
     */
    float ClassicSOM::trainStep(std::vector<float> *non_normalised_input_pattern)
    {
        std::vector<float> input_pattern = preTrainSteps(non_normalised_input_pattern);

        SOMs::GenericNeuron *winner = getWinner(&input_pattern);

        learning_rate.push_back(params->classicSOM_initial_learning_rate / (1 + iteration/params->classicSOM_constant_iterations));
        sigma.push_back(params->classicSOM_initial_sigma / (1 + iteration/params->classicSOM_constant_iterations));
        float error = 0;
        if ((learning_rate.back()!=0) && (sigma.back()!=0))
        {
            for (int i = 0; i < params->som_size; i++)
            {
                for (int j = 0; j < params->som_size; j++)
                {
                    // clear the incoming activations to this node
                    neurons[i][j]->propagated_activation=0;

                    float neigh_function = neighborhood_function(&input_pattern, winner, neurons[i][j]);
                    error += std::abs( ((classicSOM::Neuron*)neurons[i][j])->updateWeights(&input_pattern, learning_rate.back(), neigh_function, &features_min, &features_max));
                }
            }
        }
        postTrainSteps(&input_pattern);
        return error;
    }

    /**
     * @brief ClassicSOM::neighborhood_function
     * Return the value from the neighborhood function for the SOM update (see Kohonen map algorithm)
     * @param input_pattern the input pattern
     * @param winner the closest node to the input pattern
     * @param current a node in the map
     * @return the output of the function to use in the update rule of the SOM algorithm
     */
    float ClassicSOM::neighborhood_function(std::vector<float> *input_pattern, SOMs::GenericNeuron * winner, SOMs::GenericNeuron *current)
    {
        float squared_distance_in_lattice =
                lookup_squared_distances_in_lattice[winner->X][winner->Y][current->X][current->Y];

        return std::exp(-0.5*squared_distance_in_lattice/(M_PI*pow(sigma.back(),2)));
    }

}

/*
 * This is an implementation of the Dynamic DSOM algorithm
 * Nicolas Rougier and Yann Boniface, "Dynamic Self-Organising Map", Neurocomputing 74, 11 (2011), pp. 1840-1847
 *
 * DSOM algorithm is essentially a variation of the DSOM algorithm where the time dependency has been removed.
 *
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
 *
*/

#include "DSOM.h"

namespace dsom
{

    DSOM::DSOM(std::string filename)
        : GenericSOM(filename)
    {
        number_of_visited_nodes.clear();
//        neuron_activation_entropy.clear();
        loadFromFile(filename);
        init_lookup_tables();
    }

    // dim = dimension of the weights of the neurons
    // siz = size of the DSOM (siz x siz)
    DSOM::DSOM(SOMs::ParamSOM* _params)
    {
        params = _params;
        number_of_visited_nodes.clear();
//        neuron_activation_entropy.clear();
        // initialise weights randomly between -1 and 1
        std::vector<float> ranges_min(params->weights_size, -1.0);
        std::vector<float> ranges_max(params->weights_size, 1.0);

        init(&ranges_min, &ranges_max);
//        init();
        init_lookup_tables();

        // suppose the data has 0 mean and unitary stddev in each dimension
        means.resize(params->weights_size, 0.0);
        stddevs.resize(params->weights_size, 1.0);

    }

    DSOM::DSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max)
    {
        params = _params;
        number_of_visited_nodes.clear();
//        neuron_activation_entropy.clear();
        init(ranges_min, ranges_max);
        init_lookup_tables();

        // suppose the data has 0 mean and unitary stddev in each dimension
        means.resize(params->weights_size, 0.0);
        stddevs.resize(params->weights_size, 1.0);
    }

    DSOM::DSOM(SOMs::ParamSOM* _params, std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddevs)
    {
        params = _params;
        number_of_visited_nodes.clear();
//        neuron_activation_entropy.clear();
        init(ranges_min, ranges_max, _means, _stddevs);
        init_lookup_tables();
    }

    DSOM::DSOM(const SOMs::GenericSOM& gsom)
    {
        params = gsom.params;
        number_of_visited_nodes.clear();
//        neuron_activation_entropy.clear();
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
                neurons[i][j] = new dsom::Neuron(*gsom.neurons[i][j]);
            }
        }
        init_lookup_tables();
    }


    void DSOM::init()
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
                neurons[i][j] = new dsom::Neuron();
                ((dsom::Neuron*)neurons[i][j])->init(i,j,params);
            }
        }
    }

    void DSOM::init(std::vector<float> *ranges_min, std::vector<float> *ranges_max)
    {
        if (ranges_min->size()!=params->weights_size || ranges_max->size()!=params->weights_size)
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
                neurons[i][j] = new dsom::Neuron();
                ((dsom::Neuron*)neurons[i][j])->init(i,j,params, ranges_min, ranges_max);
            }
        }
    }

    void DSOM::init(std::vector<float> *ranges_min, std::vector<float> *ranges_max, std::vector<float> *_means, std::vector<float> *_stddev)
    {
        if (ranges_min->size()!=params->weights_size || ranges_max->size()!=params->weights_size || _means->size()!=params->weights_size || _stddev->size()!=params->weights_size)
            std::cerr<<"ranges_min->size()!=weight_dimensions || ranges_max->size()!=weight_dimensions || means->size()!=weight_dimensions || stddev->size()!=weight_dimensions"<<std::endl;

        iteration = 0;

        means.resize(params->weights_size);
        stddevs.resize(params->weights_size);
        for (unsigned int i=0; i< _means->size(); i++)
        {
            means[i] = _means->at(i);
            stddevs[i] = _stddev->at(i);
        }

        features_min.resize(params->weights_size);
        features_max.resize(params->weights_size);
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

        std::vector<float> zero_means;
        zero_means.resize(params->weights_size, 0.0);
        std::vector<float> unit_stddev;
        unit_stddev.resize(params->weights_size, 1.0);


        neurons.resize(params->som_size);
        for (int i = 0; i < params->som_size; i++)
        {
            neurons[i].resize(params->som_size);
            for (int j = 0; j < params->som_size; j++)
            {
                neurons[i][j] = new dsom::Neuron();
                if (params->initialise_node_weights_with_zero_means_and_unit_stddev)
                    ((dsom::Neuron*)neurons[i][j])->init(i,j,params, ranges_min, ranges_max, &zero_means, &unit_stddev);
                else
                    ((dsom::Neuron*)neurons[i][j])->init(i,j,params, ranges_min, ranges_max, &means, &stddevs);
            }
        }
    }

    void DSOM::loadFromFile (std::string filename)
    {
        std::cout<<"Loading DSOM from "<<filename.c_str()<<std::endl;
        std::ifstream inputfile;
        inputfile.open(filename.c_str(), std::ifstream::in);
        if (!inputfile.is_open())
            std::cerr<<"Could not open "<<filename.c_str()<<"\n";

        std::string line;
        // header (som_size weight_dimensions learning_rate elasticity)        
        getline (inputfile, line);

        getline (inputfile, line);
        std::istringstream iss_header(line);
        char c;
        iss_header>> c; // skip the # character
        iss_header>>params->som_size;
        iss_header>>params->weights_size;
        iss_header>>params->DSOM_learning_rate;
        iss_header>>params->DSOM_elasticity;
        //squared_elasticity = pow (elasticity,2);

        init();
        float val;
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        features_min.resize(params->weights_size);

        for (unsigned int i = 0; i<features_min.size(); i++)
        {
            iss_header>>features_min[i];
        }
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        features_max.resize(params->weights_size);
        for (unsigned int i = 0; i< features_max.size(); i++)
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
        for (unsigned int i = 0; i<means.size(); i++)
        {
            iss_header>>temp_mean[i];
            means[i] = temp_mean[i];
        }

        std::vector<float> temp_stddev;
        temp_stddev.resize(stddevs.size());
        getline (inputfile, line); // skip header
        getline (inputfile, line);
        iss_header.clear();
        iss_header.str(line);
        iss_header>> c; // skip the # character
        for (unsigned int i = 0; i<stddevs.size(); i++)
        {
            iss_header>>temp_stddev[i];
            stddevs[i] = temp_stddev[i];
        }

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
        }

        std::cout<<"DSOM loaded."<<std::endl;
    }

    void DSOM::saveNodes(std::string filename)
    {
        std::ofstream myfile;
        myfile.open (filename.c_str());
        myfile<<"# som_size weight_dimensions learning_rate elasticity"<<std::endl;
        myfile<<"# "<<params->som_size<<" "<< params->weights_size<<" "<< params->DSOM_learning_rate<<" "<<params->DSOM_elasticity<<std::endl;
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
            //myfile<<running_average_and_stddev[i].Mean()<<" ";
            myfile<<means[i]<<" ";
        }
        myfile<<std::endl;
        myfile<<"# std_dev "<<std::endl<<"# ";
        for (unsigned int i = 0; i<stddevs.size();i++)
        {
            //myfile<<running_average_and_stddev[i].StandardDeviation()<<" ";
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

                myfile<<(neurons[i][j]->number_of_visits)<<std::endl;
            }
        }
        myfile.close();
    }


    // Train step (only one input vector)
    float DSOM::trainStep(std::vector<float> *input_pattern)
    {
        preTrainSteps(input_pattern);
        float error = 0;
        SOMs::GenericNeuron *winner = getWinner(input_pattern);

        //updateLearningRateAndElasticity();

        // if input_pattern and winner are the same, the neighbourhood function is 0, so there is no update in all the nodes
        if (winner->weightsEqualTo(input_pattern))
            return 0;

        //float distance_btw_input_and_winner = getDistance(&winner->weights, input_pattern);

        for (int i = 0; i < params->som_size; i++)
        {
            for (int j = 0; j < params->som_size; j++)
            {
                // clear the incoming activations to this node
                neurons[i][j]->propagated_activation=0;

                float nf = neighborhood_function(input_pattern, winner, neurons[i][j]);
                error += std::abs( ((dsom::Neuron*)neurons[i][j])->updateWeights(input_pattern, nf, iteration, &features_min, &features_max));
            }
        }

        postTrainSteps(input_pattern);
        learning_rate_history.push_back(params->DSOM_learning_rate);
        elasticity_history.push_back(params->DSOM_elasticity);

    }

    float DSOM::neighborhood_function(std::vector<float> *input_pattern, SOMs::GenericNeuron * winner, SOMs::GenericNeuron *current)
    {
        // distance in the DSOM lattice between the current neuron and the winner
        float squared_distance_in_lattice = std::pow(winner->X - current->X, 2) + std::pow(winner->Y - current->Y, 2);
        //std::cout<< "squared_distance_in_lattice  "<<squared_distance_in_lattice <<" win "<<winner->X<<" "<<winner->Y<<std::endl;
        // distance in the weights space between the winner and the input pattern
        float squared_distance_input_winner = getSquaredDistance(&winner->weights, input_pattern);
        if (squared_distance_input_winner==0)
            return 0;
            //std::cerr<<"DSOM:neighborhood_function - (squared_distance between input and winner==0). It should have been already detected and neigh_func not called."<<std::endl;
        float temp = sqrt(squared_distance_in_lattice)/sqrt(squared_distance_input_winner);
        if (params->DSOM_elasticity == 0)
            std::cerr<<"DSOM::neighborhood_function:(elasticity == 0) "<<std::endl;
        return std::exp(-temp/pow(params->DSOM_elasticity,2));
    }

    void DSOM::updateLearningRateAndElasticity()
    {
//        if ( (collected_samples.size()>0) && (der_prediction_error.size()>0) )
//        {
//            elasticity += 0.01;
//            elasticity_history.push_back(elasticity);
//        }
//        if ( (collected_samples.size()>0) && (der_prediction_error.size()<0) )
//        {
//            elasticity -= 0.01;
//            elasticity_history.push_back(elasticity);
//        }
//        std::cout<<"DSOM elasticity "<<elasticity<<std::endl;

//        //if ( (collected_samples.size()>0) && (winner->derivative_prediction_errors.size()>0) )
//        if ( (collected_samples.size()>0) && (der_moving_quantization_error.size()>0) )
//        {
//            //std::cout<<"der_variance "<<der_variance_of_node_visits.back()<<std::endl;
//            //std::cout<<"der_dist "<<der_moving_distortion_error.back()<<std::endl;
//            //if (der_variance_of_node_visits.back()>0)
//            //if (winner->derivative_prediction_errors.back()>0)
//            if (der_moving_quantization_error.back()<0)
//            {
//                //elasticity += 0.01 * winner->derivative_prediction_errors.back();
//                learning_rate += 0.01;// * der_moving_distortion_error.back();
////                        elasticity += 0.01;
//                if (learning_rate>10)
//                    learning_rate = 10;
//                //learning_rate +=0.01 * winner->derivative_prediction_errors.back();
////                        learning_rate +=0.01 * der_moving_distortion_error.back();
//////                        learning_rate +=0.01;
////                        if (learning_rate > 3)
////                            learning_rate = 3;
//            }
//            //if (winner->derivative_prediction_errors.back()< 0)
//            if (der_moving_quantization_error.back()>0)
//            {
//                //elasticity -= 0.01 * winner->derivative_prediction_errors.back();
//                learning_rate -= 0.01;// * der_moving_distortion_error.back();
////                        elasticity -= 0.01;
//                if (learning_rate<0.5)
//                    learning_rate = 0.5;
//                //learning_rate -=0.01 * winner->derivative_prediction_errors.back();
////                        learning_rate -=0.01 * der_moving_distortion_error.back();
//////                        learning_rate -=0.01;
////                        if (learning_rate < 0.01)
////                            learning_rate = 0.01;
//            }
//            std::cout<<"learning_rate "<<learning_rate<<std::endl;
//        }
    }



    void DSOM::savePredictionErrors(std::string filename)
    {
        std::cout<<"DSOM::savePredictionErrors(std::string filename) NOT IMPLEMENTED"<<std::endl;
//        // save prediction_errors
//        std::string filename1;
//        filename1.append(filename);
//        filename1.append("_errors.out");
//        std::cout<<"Saving "<<filename1.c_str()<<std::endl;

//        std::ofstream myfile1;
//        myfile1.open (filename1.c_str());
//        myfile1<<"# number_of_visited_nodes moving_distortion_error derivative_of_moving_dist_err variance_of_node_visits derivative_variance_of_node_visits elasticity learning_rate neur_activation_entropy"<<std::endl;
//        myfile1<<"# dsom_size: "<<som_size<<std::endl;
//        for (unsigned int i=0; i< number_of_visited_nodes.size(); i++)
//        {
//            if (i < number_of_visited_nodes.size())
//                myfile1<<number_of_visited_nodes.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < moving_quantization_error.size())
//                myfile1<<moving_quantization_error.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < der_moving_quantization_error.size())
//                myfile1<<der_moving_quantization_error.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < variance_of_node_visits.size())
//                myfile1<<variance_of_node_visits.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < der_variance_of_node_visits.size())
//                myfile1<<der_variance_of_node_visits.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < elasticity_history.size())
//                myfile1<<elasticity_history.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < learning_rate_history.size())
//                myfile1<<learning_rate_history.at(i)<<" ";
//            else
//                myfile1<<"0 ";

//            if (i < neuron_activation_entropy.size())
//                myfile1<<neuron_activation_entropy.at(i)<<" ";
//            else
//                myfile1<<"0 ";
//            myfile1<<std::endl;
//        }
//        myfile1.close();

//        std::cout<<"Errors saved."<<std::endl;

    }

    void DSOM::loadPredictionErrors(std::string filename)
    {
        std::cout<<"DSOM::loadPredictionErrors(std::string filename) NOT IMPLEMENTED"<<std::endl;
//        std::string filename1;
//        filename1.append(filename);
//        filename1.append("_errors.out");

//        std::ifstream inputfile;
//        inputfile.open(filename1.c_str(), std::ifstream::in);
//        if (!inputfile.is_open())
//            std::cerr<<"Could not open "<<filename1.c_str()<<"\n";

//        std::string line;
//        // header (som_size weight_dimensions learning_rate elasticity)
//        getline (inputfile, line);
//        getline (inputfile, line);

//        number_of_visited_nodes.clear();
//        moving_quantization_error.clear();
//        der_moving_quantization_error.clear();
//        variance_of_node_visits.clear();
//        der_variance_of_node_visits.clear();
//        elasticity_history.clear();
//        learning_rate_history.clear();
//        neuron_activation_entropy.clear();

//        float value;
//        while (getline (inputfile, line))
//        {
//            std::istringstream iss(line);
//            iss>>value;     number_of_visited_nodes.push_back(value);
//            iss>>value;     moving_quantization_error.push_back(value);
//            iss>>value;     der_moving_quantization_error.push_back(value);
//            iss>>value;     variance_of_node_visits.push_back(value);
//            iss>>value;     der_variance_of_node_visits.push_back(value);
//            iss>>value;     elasticity_history.push_back(value);
//            iss>>value;     learning_rate_history.push_back(value);
//            iss>>value;     neuron_activation_entropy.push_back(value);
//        }
//        inputfile.close();
    }

}

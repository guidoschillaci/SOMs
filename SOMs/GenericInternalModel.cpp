/*
 * Generic Internal Model.
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
*/

#include "SOMs/GenericInternalModel.h"

namespace SOMs
{

    // When connecting a hebbian table, check that the output map is the same as the output maps of the
    // other connected hebbian tables!
    void GenericInternalModel::connectHebbianTable(GenericHebbianTable* hebbianTable)
    {
        hebbianTables.push_back(hebbianTable);
    }

    // before using predict, propagate the input signals using each of the hebbian tables connected (or parts of them)
    // then, use the predict function of one of the hebbian tables, which takes the most activated nodes in the output map
    // or the weighted average of the activated nodes in the output map
    GenericNeuron*  GenericInternalModel::predict()
    {
        if (hebbianTables.size() == 0)
            std::cerr<<"GenericInternalModel::predict() - (hebbianTables.size() == 0)"<<std::endl;
        return hebbianTables.at(0)->predict();
    }

    // before using computePredictionError, propagate the input patterns from at least one of the connected hebbian tables
    float GenericInternalModel::computePredictionError(std::vector<float> *output_pattern)
    {
        if (hebbianTables.size() == 0)
            std::cerr<<"GenericInternalModel::computePredictionError() - (hebbianTables.size() == 0)"<<std::endl;
        return hebbianTables.at(0)->outputSOM->getDistance(output_pattern, &predict()->weights);
    }

    // compute prediction error on a set of test points (collected previously and used for measuring the quality of the learning
    void GenericInternalModel::computeMovingError()
    {
        if (hebbianTables.size() == 0)
            std::cerr<<"GenericInternalModel::computeMovingError() - (hebbianTables.size() == 0)"<<std::endl;

        float error = 0;
        for (unsigned int d=0; d<hebbianTables.at(0)->buffer_data_input.size(); d++)
        {
            hebbianTables.at(0)->outputSOM->clearIncomingActivations();
            for (unsigned int h=0; h < hebbianTables.size(); h++)
            {
                if (hebbianTables.at(h)->buffer_data_input.size() != hebbianTables.at(h)->buffer_data_output.size())
                    std::cerr<<"GenericInternalModel:computeMovingError. (buffer_data_input.size()!=buffer_data_output.size()) "<<std::endl;
                if (hebbianTables.at(h)->buffer_data_input.size()==0)
                    return;

                hebbianTables.at(h)->propagateInputToOutputMap(&hebbianTables.at(h)->buffer_data_input[d]);
                //std::cout<<"Propagation "<<h<<std::endl;
                //hebbianTables.at(h)->outputSOM->printPropagatedActivations();
            }
            float temp_err = computePredictionError( &hebbianTables.at(0)->buffer_data_output[d] );
            error += temp_err;
        }

        if (hebbianTables.at(0)->buffer_data_input.size()==0)
            std::cerr<<"GenericInternalModel::computeMovingError(): (buffer_data_input.size()==0)"<<std::endl;
        moving_error.push_back( error / float(hebbianTables.at(0)->buffer_data_input.size()) );

        //computeDerivativeOfMovingError();
    }

    void GenericInternalModel::computeDerivativeOfMovingError()
    {
        int window_size = 50;

        if (moving_error.size()<window_size)
        {
            derivative_moving_error.push_back(0);
            return;
        }
        int half_window_size=int(float(window_size)/2);
        float left_sum = 0;
        float right_sum = 0;
        for (unsigned int i=moving_error.size()-half_window_size; i<moving_error.size(); i++)
        {
            right_sum += moving_error.at(i);
        }
        for (unsigned int i=moving_error.size()-window_size; i<moving_error.size()-half_window_size; i++)
        {
            left_sum += moving_error.at(i);
        }
        derivative_moving_error.push_back((right_sum-left_sum)/float(window_size));
    }


    void GenericInternalModel::loadMovingError(std::string filename)
    {
        std::cout<<"Loading "<<filename.c_str()<<std::endl;

        std::ifstream inputfile;
        inputfile.open(filename.c_str(), std::ifstream::in);

        std::string line;
        getline (inputfile, line);

        moving_error.clear();
        float val;
        while(getline (inputfile, line))
        {
            std::istringstream iss(line);
            iss>>val; moving_error.push_back(val);
            //iss>>val; derivative_moving_error.push_back(val);
        }
        inputfile.close();
    }


    void GenericInternalModel::saveMovingError(std::string filename)
    {
        std::cout<<"Saving "<<filename.c_str()<<std::endl;
        std::ofstream myfile;
        myfile.open (filename.c_str());
        //myfile<<"# moving_error derivative_of_moving_error"<<std::endl;
        myfile<<"# moving_error "<<std::endl;
//        if (moving_error.size() != derivative_moving_error.size())
//            std::cerr<<"GenericInternalModel::saveMovingError: (moving_error.size() != derivative_moving_error.size()). "<<filename.c_str()<<std::endl;
        for (unsigned int i=0; i<moving_error.size(); i++)
        {
            myfile<<moving_error[i]<<" ";
            //myfile<<derivative_moving_error[i]<<" ";
            myfile<<std::endl;
        }
        myfile.close();
    }

}


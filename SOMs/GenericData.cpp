/*
 * This is a generic class for a Neuron
 *
 *
 * Author: guido schillaci
 * Humboldt-Universitaet zu Berlin
 * email: guido.schillaci@informatik.hu-berlin.de
 *
*/

#include "SOMs/GenericData.h"

namespace SOMs
{

GenericData::GenericData(std::vector<float> *in)
{
    data = *in;
    is_normalised = false;
}

GenericData::GenericData(std::vector<float> *in, int _normalisation_type, std::vector<float> *_features_min, std::vector<float> *_features_max, std::vector<RunningStat> *_running_avg_and_stddev)
{
//    for (unsigned int i=0; i<in->size(); i++)
//        data.push_back(in->at(i));
    data.resize(in->size());
    normalisation_type = _normalisation_type;

    features_min = _features_min;
    features_max = _features_max;
    running_avg_and_stddev = _running_avg_and_stddev;

    normalise(normalisation_type, features_min, features_max, running_avg_and_stddev);

    is_normalised = true;

}


GenericData& GenericData::operator=(GenericData const &d)
{
    data.resize(d.data.size());
    normalised_data.resize(d.normalised_data.size());
    if (data.size() != normalised_data.size())
        std::cout<<"error: GenericData::operator= (data.size() != normalised_data.size())"<<std::endl;

    for (unsigned int i=0; i<data.size(); i++)
    {
        data[i] = d.data[i];
        normalised_data[i] = d.normalised_data[i];
    }

    normalisation_type = d.normalisation_type;
    features_min = d.features_min;
    features_max = d.features_max;
    running_avg_and_stddev = d.running_avg_and_stddev;
}


    void GenericData::normalise(int _normalisation_type, std::vector<float> *_features_min, std::vector<float> *_features_max, std::vector<RunningStat> *_running_avg_and_stddev)
    {
        normalisation_type = _normalisation_type;
        features_min = _features_min;
        features_max = _features_max;
        running_avg_and_stddev = _running_avg_and_stddev;

        if (normalisation_type == NORMALISE_ACTIV_WITH_FEAT_SCALING)
        {
            for (unsigned int i=0; i<data.size(); i++)
            {
                float den = features_max->at(i) - features_min->at(i);
                if ( den <= 0)
                    std::cerr<<"GenericData (features_max - features_min <= 0)"<<std::endl;
                normalised_data.push_back( (data.at(i) - features_min->at(i)) / den);
            }
        }
        else // (normalisation_type == NORMALISE_ACTIV_WITH_RUNNING_AVG_STDDEV)
        {
            for (unsigned int i=0; i<data.size(); i++)
            {
                float mean=running_avg_and_stddev->at(i).Mean();
                float stddev=running_avg_and_stddev->at(i).StandardDeviation();
                if (stddev == 0)
                    std::cerr<<"genericdata error: (stddev == 0)"<<std::endl;
                normalised_data.push_back( (data[i]-mean) / stddev );
            }
        }
        is_normalised = true;
    }

}

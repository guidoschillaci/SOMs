/**
 * @author Guido Schillaci - Humboldt Universität zu Berlin
 */

#include "SOMs/ClassicSOM/Neuron.h"
#include "SOMs/ClassicSOM/ClassicSOM.h"
#include "SOMs/ClassicSOM/HebbianTable.h"

#include "SOMs/ParamSOM.h"
#include "SOMs/ParamHebbianTable.h"


SOMs::ParamHebbianTable     param_hebbian;
SOMs::ParamSOM params_SOM_A;
SOMs::ParamSOM params_SOM_B;
classicSOM::ClassicSOM      *SOM_A;
classicSOM::ClassicSOM      *SOM_B;

classicSOM::HebbianTable    *HebbTable_AB;


bool initialiseModels()
{

    // initialise SOMs
    params_SOM_A.som_size = 10; // lattice space
    params_SOM_A.weights_size= 2; // feature space

    params_SOM_B.som_size = 10; // lattice space
    params_SOM_B.weights_size= 2; // feature space

    // if you know already the min and max values of each dimension in the feature space, and their statistics     
    // (means and stddevs of each dimension), use the following constructor:
    // the size of min_vals, max_vals, means and stddevs should match params_SOM_A.weights_size
    // SOM_A = new classicSOM::ClassicSOM(&params_SOM_A, &min_vals, &max_vals, &means, &stddevs);
    SOM_A = new classicSOM::ClassicSOM(&params_SOM_A);
    SOM_B = new classicSOM::ClassicSOM(&params_SOM_B);

    // initialise hebbian table (directional links from SOM_A to SOM_B
    param_hebbian.hebbian_learning_rate = 0.01;
    param_hebbian.propagation_type = PROPAGATE_WINNER_TO_ALL;
    HebbTable_AB = new classicSOM::HebbianTable(&param_hebbian, SOM_A, SOM_B);
    HebbTable_AB->param->printParams();
}

// example of training the function B = A
// it requires that SOM_A->params->weights_size == SOM_B->params->weights_size
bool trainModels()
{

    std::vector<float> input_A(SOM_A->params->weights_size, 0.0);
    std::vector<float> input_B(SOM_B->params->weights_size, 0.0);

    for (int i=0; i< 50000; i++)
    {
        // train the SOMs with random values sampled from a uniform distribution
        for (unsigned int j=0; j< input_A.size(); j++)
        {
            input_A.at(j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
        }
        for (unsigned int j=0; j< input_B.size(); j++)
        {
            input_B.at(j) = input_A.at(j);
        }


        // TRAIN THE SOMs
        SOM_A->trainStep(&input_A);
        SOM_B->trainStep(&input_B);
        // update the hebbian table. The table is uni-directional: from A (input) to B (output)
        HebbTable_AB->update(&input_A, &input_B, 1); // 1 is an additional factor that can be multiplied to the update rule


        // print out some sample predictions
        std::vector<float> query_1(SOM_A->params->weights_size, 0.0);

        for (unsigned int k=0; k<SOM_A->params->weights_size; k++)
        {
            query_1.at(k) = 0.5;
        }

        // clear the output SOM from any activation that has been propagated through the hebbian table
        SOM_B->clearIncomingActivations();
        HebbTable_AB->propagateInputToOutputMap(&query_1, true); // true: normalise input; false: do not normalise input
        std::vector<float> output_1 = HebbTable_AB->predict()->weights;
        std::cout<<"Iteration "<<i<<std::endl;
        std::cout<<"input 1: (";
        for (unsigned int k=0; k<SOM_A->params->weights_size; k++)
        {
            std::cout<<query_1.at(k)<<", ";
        }
        std::cout<<"); prediction: (";
        for (unsigned int k=0; k<SOM_A->params->weights_size; k++)
        {
            std::cout<<output_1.at(k)<<", ";
        }
        std::cout<<")"<<std::endl;

        //////////////////////

    }

}



int main(int argc, char* argv[])
{

    int seed = time(NULL);
    srand(seed);

    std::cout<<"Initialising models..."<<std::endl;
    initialiseModels();
    std::cout<<"...models initialised!"<<std::endl;

    trainModels();

//    SOM_A->printNodes();

}








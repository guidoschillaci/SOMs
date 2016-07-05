/*! @file
* @author Guido Schillaci - Humboldt Universität zu Berlin <guido.schillaci@informatik.hu-berlin.de>
*/


#include "SOMs/GenericHebbianTable.h"

namespace SOMs
{

    /**
     * @brief GenericHebbianTable::GenericHebbianTable
     * Create an Generic Hebbian table, with the parameters specified in param and with input map and output map specified
     * The axons are initialised to 0
     * @param param the parameters of the hebbian table
     * @param input_s the input som
     * @param output_s the output som
     */
    GenericHebbianTable::GenericHebbianTable(ParamHebbianTable *param, GenericSOM *input_s, GenericSOM *output_s)
    {
        if ((param->hebbian_rule_type != USE_OJA_RULE) && (param->hebbian_rule_type != USE_POS_HEBBIAN_RULE))
            std::cerr<<"Error in defining GenericHebbianTable. Set the hebbian_rule_type parameter as: USE_OJA_RULE or USE_POS_HEBBIAN_RULE"<<std::endl;

        if ((param->learning_strategy != LEARN_USING_WINNERS)
                && (param->learning_strategy != LEARN_USING_WINNERS_AND_NEIGHBOURS)
                && (param->learning_strategy != LEARN_USING_ALL))
            std::cerr<<"Error in defining GenericHebbianTable. Set the learning_type parameter as: LEARN_USING_WINNERS, LEARN_USING_WINNERS_AND_NEIGHBOURS or LEARN_USING_ALL"<<std::endl;        

        this->param = param;

        inputSOM = input_s;
        outputSOM = output_s;

        axons.resize(inputSOM->params->som_size);
        for (unsigned int i=0; i<inputSOM->params->som_size; i++)
        {
            axons[i].resize(inputSOM->params->som_size);
            for (unsigned int j=0; j<inputSOM->params->som_size; j++)
            {
                axons[i][j].resize(outputSOM->params->som_size);
                for (unsigned int k=0; k<outputSOM->params->som_size; k++)
                {
                    axons[i][j][k].resize(outputSOM->params->som_size);
                    for (unsigned int q=0; q<outputSOM->params->som_size; q++)
                    {
                      axons[i][j][k][q]=0;
                    }
                }
            }
        }
    }

    /**
     * @brief GenericHebbianTable::connectSOMs
     * Set the input and the output maps. This is already done in the constructor but should be called after loadTable
     * @param i_s the input map
     * @param o_s the output map
     */
    void GenericHebbianTable::connectSOMs(GenericSOM *i_s, GenericSOM *o_s)
    {
        inputSOM = i_s;
        outputSOM= o_s;
    }

    /**
     * @brief GenericHebbianTable::update
     * this is the function for updating the hebbian table using the input patterns. It calls learnUsing*, according to the strategy selected in ParamHebbianTable
     * @param non_normalised_input_pattern a non normalised input pattern (will be normalised within the function)
     * @param non_normalised_output_pattern  a non normalised output pattern (will be normalised within the function)
     * @param additionalFactor do you want to multiply an additional factor to the hebbian update rule?
     */
    void GenericHebbianTable::update(std::vector<float> *non_normalised_input_pattern, std::vector<float> *non_normalised_output_pattern, float additionalFactor = 1)
    {
        std::vector<float> input_pattern = inputSOM->normalise_input(non_normalised_input_pattern);
        std::vector<float> output_pattern = outputSOM->normalise_input(non_normalised_output_pattern);

        // select the learning strategy to use
        if (param->learning_strategy == LEARN_USING_WINNERS)
            learnUsingWinners(&input_pattern, &output_pattern, additionalFactor);
        if (param->learning_strategy == LEARN_USING_WINNERS_AND_NEIGHBOURS)
            learnUsingWinnersAndNeighbors(&input_pattern, &output_pattern, additionalFactor);
        if (param->learning_strategy == LEARN_USING_ALL)
            learnUsingAll(&input_pattern, &output_pattern, additionalFactor);

        if (param->normalise_hebbian_weights)
            normaliseWeights();

    }

    /**
     * @brief GenericHebbianTable::learnUsingWinners
     * An update strategy which only updates the connection of the winner in one map to the winner in the other map
     * @param input_pattern the input pattern (should be already normalised)
     * @param output_pattern  the output pattern (should be already normalised)
     * @param additionalFactor do you want to multiply an additional factor to the hebbian update rule?
     */
    void GenericHebbianTable::learnUsingWinners(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor)
    {
        GenericNeuron *input_winner = inputSOM->getWinner(input_pattern);
        GenericNeuron *output_winner = outputSOM->getWinner(output_pattern);
        if ((input_winner->X >= inputSOM->params->som_size)||(input_winner->Y>= inputSOM->params->som_size) ||
                (output_winner->X >= outputSOM->params->som_size)||(output_winner->Y >= outputSOM->params->som_size) )
            std::cerr<<"Error in GenericHebbian table-> learnUsingWinners. Winner's X or Y > of the som_size"<<std::endl;

        input_winner->number_of_visits++;
        output_winner->number_of_visits++;

        float a_x, a_y;
        a_x = inputSOM->neurons[input_winner->X][input_winner->Y]->getActivation(input_pattern, &inputSOM->features_min, &inputSOM->features_max);
        a_y = outputSOM->neurons[output_winner->X][output_winner->Y]->getActivation(output_pattern, &outputSOM->features_min, &outputSOM->features_max);

        if (param->hebbian_rule_type == USE_POS_HEBBIAN_RULE)
            axons[input_winner->X][input_winner->Y][output_winner->X][output_winner->Y] += param->hebbian_learning_rate * a_x * a_y * additionalFactor;

        if (param->hebbian_rule_type == USE_OJA_RULE)
        {
            axons[input_winner->X][input_winner->Y][output_winner->X][output_winner->Y] +=
                    param->hebbian_learning_rate * a_y * ( a_x - a_y * axons[input_winner->X][input_winner->Y][output_winner->X][output_winner->Y]);
        }
    }

    /**
     * @brief GenericHebbianTable::learnUsingWinnersAndNeighbors
     * update the connection of the winner and of its neighbours in one map to the winner and of its neighbours in the other map
     * @param input_pattern the input pattern (should be already normalised)
     * @param output_pattern  the output pattern (should be already normalised)
     * @param additionalFactor do you want to multiply an additional factor to the hebbian update rule?
     */
    void GenericHebbianTable::learnUsingWinnersAndNeighbors(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor)
    {
        GenericNeuron *input_winner = inputSOM->getWinner(input_pattern);
        GenericNeuron *output_winner = outputSOM->getWinner(output_pattern);

        input_winner->number_of_visits++;
        output_winner->number_of_visits++;

        int start_x1, start_y1;
        int start_x2, start_y2;
        int end_x1, end_y1;
        int end_x2, end_y2;

        if (input_winner->X-param->neighbours_size >= 0)
            start_x1 = input_winner->X -param->neighbours_size;
        else
            start_x1 = 0;
        if (input_winner->X+param->neighbours_size < inputSOM->params->som_size)
            end_x1 = input_winner->X+param->neighbours_size;
        else
            end_x1 = inputSOM->params->som_size - 1;

        if (input_winner->Y-param->neighbours_size >= 0)
            start_y1 = input_winner->Y -param->neighbours_size;
        else
            start_y1 = 0;
        if (input_winner->Y+param->neighbours_size < inputSOM->params->som_size)
            end_y1 = input_winner->Y+param->neighbours_size;
        else
            end_y1 = inputSOM->params->som_size - 1;

        if (output_winner->X-param->neighbours_size >= 0)
            start_x2 = output_winner->X -param->neighbours_size;
        else
            start_x2 = 0;
        if (output_winner->X+param->neighbours_size < outputSOM->params->som_size)
            end_x2 = output_winner->X+param->neighbours_size;
        else
            end_x2 = outputSOM->params->som_size - 1;

        if (output_winner->Y-param->neighbours_size >= 0)
            start_y2 = output_winner->Y -param->neighbours_size;
        else
            start_y2 = 0;
        if (output_winner->Y+param->neighbours_size < outputSOM->params->som_size)
            end_y2 = output_winner->Y+param->neighbours_size;
        else
            end_y2 = outputSOM->params->som_size - 1;

        for (unsigned int x1 = start_x1; x1<=end_x1; x1++)
        {
            for (unsigned int y1 = start_y1; y1<=end_y1; y1++)
            {
                for (unsigned int x2 = start_x2; x2<=end_x2; x2++)
                {
                    for (unsigned int y2 = start_y2; y2<=end_y2; y2++)
                    {
                        float a_x, a_y;
                        a_x = inputSOM->neurons[x1][y1]->getActivation(input_pattern, &inputSOM->features_min, &inputSOM->features_max);
                        a_y = outputSOM->neurons[x2][y2]->getActivation(output_pattern, &outputSOM->features_min, &outputSOM->features_max);

                        if ((a_x!=0) && (a_y!=0))
                        {
                            if (param->hebbian_rule_type == USE_POS_HEBBIAN_RULE)
                                axons[x1][y1][x2][y2]+= param->hebbian_learning_rate * a_x * a_y * additionalFactor;
                            if (param->hebbian_rule_type == USE_OJA_RULE)
                            {
                                axons[x1][y1][x2][y2] +=
                                        param->hebbian_learning_rate * a_y * ( a_x - a_y * axons[x1][y1][x2][y2]);
                            }
                        }

                    }
                }
            }
        }
    }


    /**
     * @brief GenericHebbianTable::learnUsingAll
     * update all the connections very slow strategy and not very well tested. Use LEARN_USING_WINNERS, instead
     * @param input_pattern the input pattern (should be already normalised)
     * @param output_pattern  the output pattern (should be already normalised)
     * @param additionalFactor do you want to multiply an additional factor to the hebbian update rule?
     */
    void GenericHebbianTable::learnUsingAll(std::vector<float> *input_pattern, std::vector<float> *output_pattern, float additionalFactor)
    {


        for (unsigned int x1 = 0; x1<inputSOM->params->som_size; x1++)
        {
            for (unsigned int y1 = 0; y1<inputSOM->params->som_size; y1++)
            {
                float a_x = inputSOM->neurons[x1][y1]->getActivation(input_pattern, &inputSOM->features_min, &inputSOM->features_max);

                if ( !( (a_x==0) && (param->hebbian_rule_type == USE_POS_HEBBIAN_RULE) ) )
                {
                    for (unsigned int x2 = 0; x2<outputSOM->params->som_size; x2++)
                    {
                        for (unsigned int y2 = 0; y2<outputSOM->params->som_size; y2++)
                        {
                            float a_y = outputSOM->neurons[x2][y2]->getActivation(output_pattern, &outputSOM->features_min, &outputSOM->features_max);

                            if (a_y!=0)
                            {
                                if (param->hebbian_rule_type == USE_POS_HEBBIAN_RULE)
                                    axons[x1][y1][x2][y2] += param->hebbian_learning_rate * a_x * a_y * additionalFactor;
                                if (param->hebbian_rule_type == USE_OJA_RULE)
                                {
                                    axons[x1][y1][x2][y2] = param->hebbian_learning_rate * a_y * ( a_x - a_y * axons[x1][y1][x2][y2]);
                                }
                                if (axons[x1][y1][x2][y2]<0)
                                    axons[x1][y1][x2][y2] = 0;
                            }
                        }
                    }
                }
            }
        }
    }



    /**
     * @brief GenericHebbianTable::getKActivatedOutputNodes
     * returns a list of the k best activated nodes on the output map, selected from the connections from input_winner in inputSOM.
     * it returns also a list of normalised activations for the k neurons
     * @param k the number of nodes to take
     * @param input_winner the pointer to the winner node in the input map
     * @param activatedNodes list of the k best activated nodes on output
     * @param activations a list of normalised activations for the k neurons
     */
    void GenericHebbianTable::getKActivatedOutputNodes(int k, GenericNeuron *input_winner,
                                                 std::vector<GenericNeuron*> *activatedNodes,
                                                 std::vector<float> *activations)
    {
        if (k<=0)
            std::cerr<<"GenericHebbianTable:getKActivatedOutputNodes, k<=0"<<std::endl;

        // it collects the connections, putting always the strongest one as the first of the list
        std::priority_queue<std::pair<float, int> > queue;

        float sum_activations=0;
        for (int i = 0; i < outputSOM->params->som_size; i++)
        {
            for (int j = 0; j < outputSOM->params->som_size; j++)
            {
                if (axons[input_winner->X][input_winner->Y][i][j] != 0)
                {
                    queue.push(std::pair<float, int>(axons[input_winner->X][input_winner->Y][i][j], j+i*outputSOM->params->som_size));
                    sum_activations+=axons[input_winner->X][input_winner->Y][i][j];
                }
            }
        }

        // if all the connections were 0, then choose a random node
        if (queue.size()==0)
        {
            int i = Utils::random(0,outputSOM->params->som_size-1);
            int j = Utils::random(0,outputSOM->params->som_size-1);
            activatedNodes->push_back(outputSOM->neurons[i][j]);
            activations->push_back(0);
            return;
        }


        // take the node with the strongest connection
        for (int i = 0; i < queue.size(); i++) {
          int ki = queue.top().second;
          int x=ki/outputSOM->params->som_size;
          int y=ki%outputSOM->params->som_size;
          activatedNodes->push_back(outputSOM->neurons[x][y]);
          activations->push_back(axons[input_winner->X][input_winner->Y][x][y] / sum_activations);
          queue.pop();
        }
    }

    /**
     * @brief GenericHebbianTable::getBestActivatedOutputNode
     * From the node in the input map specified by input_winner, selects the node
     * in the output map with the strongest connection
     * @param input_winner the pointer to the node in the input map
     * @return the pointer of the node in the output map
     */
    GenericNeuron * GenericHebbianTable::getBestActivatedOutputNode(GenericNeuron *input_winner)
    {
        float current_strongest_weight = -1;
        int best_x = 0;
        int best_y = 0;
        std::vector<int> winners_x;
        std::vector<int> winners_y;

        for (unsigned int x = 0; x < outputSOM->params->som_size; x++)
        {
            for (unsigned int y = 0; y < outputSOM->params->som_size; y++)
            {
//                if ((x==0) && (y==0))
//                {
//                    current_strongest_weight = axons[input_winner->X][input_winner->Y][x][y];
//                    winners_x.push_back(x);
//                    winners_y.push_back(y);
//                }
//                else
                {
                    if (axons[input_winner->X][input_winner->Y][x][y] > current_strongest_weight)
                    {
                        current_strongest_weight = axons[input_winner->X][input_winner->Y][x][y];
                        best_x = x;
                        best_y = y;
                        winners_x.clear();
                        winners_y.clear();
                        winners_x.push_back(best_x);
                        winners_y.push_back(best_y);
                    }
                    else if (axons[input_winner->X][input_winner->Y][x][y] == current_strongest_weight)
                    {
                        winners_x.push_back(x);
                        winners_y.push_back(y);
                    }
                }

            }
        }
        // if all the connection were 0, then returns an empty node
        if (current_strongest_weight==0)
            return new GenericNeuron(-1,-1, outputSOM->params);

        // if you have more connections with the same weight>0, then choose one randomly
        int best_index = 0;
        if (winners_x.size()>1)
        {
            best_index = rand() % (int) winners_x.size(); // avoid taking always the last element
        }
        return outputSOM->neurons[winners_x[best_index] ][winners_y[best_index]];
    }

    /**
     * @brief GenericHebbianTable::getBestActivatedOutputNode
     * From the closest node in the input map to the specified input pattern, selects the node
     * in the output map with the strongest connection
     * @param input_pattern the (normalised) input pattern
     * @return the pointer of the node in the output map
     */
    GenericNeuron * GenericHebbianTable::getBestActivatedOutputNode(std::vector<float> *input_pattern)
    {
        GenericNeuron *input_winner = inputSOM->getWinner(input_pattern);
        return getBestActivatedOutputNode(input_winner);
    }

    /**
     * @brief GenericHebbianTable::getBestActivatedOutputNode
     * From the node in the input map specified by the coordinates in the lattice, selects the node
     * in the output map with the strongest connection
     * @param node_x the x coordinate in the lattice of the input map
     * @param node_y the y coordinate in the lattice of the input map
     * @return the pointer to the best node in the output map
     */
    GenericNeuron * GenericHebbianTable::getBestActivatedOutputNode(int node_x, int node_y)
    {
        return getBestActivatedOutputNode(inputSOM->neurons[node_x][node_y]);
    }

    /**
     * @brief GenericHebbianTable::propagateInputToOutputMap
     * Propagate the activation of the node in the input map with lattice coordinates specified
     * to the output map
     * Use predict after all the propagation you want to perform to the output map, to get the final prediction
     * @param node_x the x coordinate in the lattice of the input map
     * @param node_y the y coordinate in the lattice of the input map
     * @return the total propagation
     */
    float GenericHebbianTable::propagateInputToOutputMap(int node_x, int node_y)
    {
        return propagateInputToOutputMap(&inputSOM->neurons[node_x][node_y]->weights);
    }


    /**
     * @brief GenericHebbianTable::propagateInputToOutputMap
     * Propagate from the input map to the output map.
     * It uses the propagation strategy specified in ParamHebbianTable.
     * It does not still outputs a prediction, since the output map can input propagations from other maps
     * Use predict after all the propagation you want to perform to the output map (maybe you want to propagate from other hebbian tables)
     * to get the final prediction
     * @param input_pattern the input pattern where the propagation in the input map starts from
     * @param normalise_input if true, it normalises the input. If false, it takes the input as it is.
     * @return the total propagation in the output map
     */
    float GenericHebbianTable::propagateInputToOutputMap(std::vector<float> *input_pattern, bool normalise_input)
    {       
        if (input_pattern->size() != inputSOM->params->weights_size)
            std::cerr<<"GenericHebbianTable:propagateInputToOutput. (input_pattern->size() != inputSOM->weight_dimensions)"<<std::endl;

        std::vector<float> normalised_input;
        std::vector<float> *input;
        if (normalise_input)
        {
            normalised_input = inputSOM->normalise_input(input_pattern);
            input = &normalised_input;
        }
        else
            input = input_pattern;

        bool goToAll2Winner = false;
        float total_propagation = 0;
        if (param->propagation_type == PROPAGATE_WINNER_TO_WINNER)
        {
            GenericNeuron* winner_in_input = inputSOM->getWinner(input);
            GenericNeuron* bestConnectedNode = getBestActivatedOutputNode(winner_in_input);

            // if winner has no connections to the other map, then propagate from all input nodes
            if (bestConnectedNode->X ==-1)
                goToAll2Winner =true;

            if (!goToAll2Winner)
            {
                float input_neuron_activation;
                input_neuron_activation = winner_in_input->getActivation(input, &inputSOM->features_min, &inputSOM->features_max);

                // propagate the activation to the best connected node in the output map
                if (input_neuron_activation > 0)
                {
                    bestConnectedNode->propagated_activation += input_neuron_activation * axons[winner_in_input->X][winner_in_input->Y][bestConnectedNode->X][bestConnectedNode->Y];
                    total_propagation = bestConnectedNode->propagated_activation ;
                }
            }
        }
        if (param->propagation_type == PROPAGATE_WINNER_TO_ALL)
        {

            // priority_queue contains a ordered list of elements: [distance_to_input, neuron]
            std::priority_queue<std::pair<float, GenericNeuron *>, std::vector<std::pair<float, GenericNeuron *> >, SOMs::CompareWinners > list_of_winners;
            inputSOM->getListOfWinners(input, &list_of_winners);

            while(list_of_winners.size()>0)
            {
                float d = list_of_winners.top().first;

                GenericNeuron* current_winner = list_of_winners.top().second;
                float input_neuron_activation;
                input_neuron_activation = current_winner->getActivation(input, &inputSOM->features_min, &inputSOM->features_max);

                // propagate the activation to the best connected node in the output map
                if (input_neuron_activation > 0)
                {
                    bool hasConnection = false;
                    for (int i = 0; i < outputSOM->params->som_size; i++)
                    {
                        for (int j = 0; j < outputSOM->params->som_size; j++)
                        {
                            if (axons[current_winner->X][current_winner->Y][i][j] > 0)
                            {

                                outputSOM->neurons[i][j]->propagated_activation += input_neuron_activation * axons[current_winner->X][current_winner->Y][i][j];
                                total_propagation += outputSOM->neurons[i][j]->propagated_activation;
                                hasConnection = true;
                            }
                        }
                    }
                    if (hasConnection)
                        return total_propagation;
                    else
                    {
                        //return 0;
                        list_of_winners.pop();
                        //continue; // check if the following winner is connected
                    }
                }
                else
                {
                    list_of_winners.pop();
                }
            }
            std::cout<<"Hebbian table: maps are not connected!"<<std::endl;
            return 0; // if it reaches this point, no connections between the two maps exist
        }

        if (param->propagation_type == PROPAGATE_K_WINNERS_TO_ALL)
        {

            // priority_queue contains a ordered list of elements: [distance_to_input, neuron]
            std::priority_queue<std::pair<float, GenericNeuron *>, std::vector<std::pair<float, GenericNeuron *> >, SOMs::CompareWinners > list_of_winners;
            inputSOM->getListOfWinners(input, &list_of_winners);

            //for (unsigned int w = 0; w < list_of_winners.size(); w++)
            for (unsigned int w = 0; w < SIZE_LIST_OF_WINNERS; w++)
            {
                float d = list_of_winners.top().first;

                GenericNeuron* current_winner = list_of_winners.top().second;
                float input_neuron_activation;
                input_neuron_activation = current_winner->getActivation(input, &inputSOM->features_min, &inputSOM->features_max);

                // propagate the activation to the best connected node in the output map
                if (input_neuron_activation > 0)
                {
                    bool hasConnection = false;
                    for (int i = 0; i < outputSOM->params->som_size; i++)
                    {
                        for (int j = 0; j < outputSOM->params->som_size; j++)
                        {
                            if (axons[current_winner->X][current_winner->Y][i][j] > 0)
                            {
                                outputSOM->neurons[i][j]->propagated_activation += input_neuron_activation * axons[current_winner->X][current_winner->Y][i][j];
                                total_propagation += outputSOM->neurons[i][j]->propagated_activation;
                                hasConnection = true;
                            }
                        }
                    }
                    if (hasConnection)
                        return total_propagation;
                    else
                    {
                        list_of_winners.pop();
                    }
                }
                else
                {
                    list_of_winners.pop();
                }
            }


        }
        if (param->propagation_type == PROPAGATE_ALL_TO_WINNERS)
        {
          //  std::cout<<"PROPAGATE_ALL_TO_WINNERS"<<std::endl;
            for (int i = 0; i < inputSOM->params->som_size; i++)
            {
                for (int j = 0; j < inputSOM->params->som_size; j++)
                {
                    GenericNeuron* bestConnectedNode = getBestActivatedOutputNode(inputSOM->neurons[i][j]);

                    // if the input node has no connection to the output map, then don't propagate it.
                    if (bestConnectedNode->X!=-1)
                    {
                        float input_neuron_activation;
                        input_neuron_activation = inputSOM->neurons[i][j]->getActivation(input, &inputSOM->features_min, &inputSOM->features_max);

                        // propagate the activation to the best connected node in the output map
                        if (input_neuron_activation > 0)
                        {
                            bestConnectedNode->propagated_activation += input_neuron_activation * axons[i][j][bestConnectedNode->X][bestConnectedNode->Y];
                            total_propagation +=bestConnectedNode->propagated_activation ;
                        }
                    }
                }
            }
        }
        if (param->propagation_type == PROPAGATE_ALL_TO_ALL)
        {
            for (int ix = 0; ix < inputSOM->params->som_size; ix++)
            {
                for (int iy = 0; iy < inputSOM->params->som_size; iy++)
                {
                    float input_neuron_activation;
                    input_neuron_activation = inputSOM->neurons[ix][iy]->getActivation(input, &inputSOM->features_min, &inputSOM->features_min);

                    // propagate the activation to the best connected node in the output map
                    if (input_neuron_activation > 0)
                    {
                        for (int ox = 0; ox < outputSOM->params->som_size; ox++)
                        {
                            for (int oy = 0; oy < outputSOM->params->som_size; oy++)
                            {
                                outputSOM->neurons[ox][oy]->propagated_activation += input_neuron_activation * axons[ix][iy][ox][oy];
                                total_propagation +=outputSOM->neurons[ox][oy]->propagated_activation ;
                            }
                        }
                    }
                }
            }
        }
        return total_propagation;
    }

    /**
     * @brief GenericHebbianTable::predict
     * This has to be run after having propagated at least once to the output map (also from other hebbian tables)
     * It computes the prediction with the strategy specified in ParamHebbianTable.prediction_type
     * @return a pointer to a neuron with the weights set to the predicted pattern. It could not be pointing to any actual neuron in the output map
     */
    GenericNeuron*  GenericHebbianTable::predict()
    {
        if (param->prediction_type == PREDICTION_TYPE_SIMPLE)
            return predict_simple();
        else if (param->prediction_type == PREDICTION_TYPE_WEIGHTED_AVERAGE)
            return predict_weightedAverage();
        else
        {
            std::cerr<<"Prediction type not set properly in ParamHebbianTable."<<std::endl;
            return NULL;
        }
    }

    /**
     * @brief GenericHebbianTable::predictAndUnnormalise
     * Runs predict and then unnormalise the data, suing the normalisation strategy specified
     * in the ParamSOM of the outputmap
     * @return a pointer to a neuron with the weights set to the predicted pattern. It could not be pointing to any actual neuron in the output map
     */
    GenericNeuron*  GenericHebbianTable::predictAndUnnormalise()
    {
        GenericNeuron* predictedNode = new GenericNeuron(-1, -1, outputSOM->params);
        predictedNode = predict();
        if (outputSOM->params->normalisation_type == NORMALISE_WEIGHTS_WITH_STANDARD_SCORE)
        {
            for (unsigned int i=0; i<predictedNode->weights.size(); i++)
            {
                predictedNode->weights.at(i) = (predictedNode->weights.at(i) * outputSOM->stddevs[i] )
                        + outputSOM->means[i];
            }
        }
        return predictedNode;
    }


    /**
     * @brief GenericHebbianTable::predict_simple
     * Compute the prediction in the output map as the best activated node (activations are computed from propagated signals using propagateInputToOutputMap)
     * @return  a pointer to the most activated neuron in the output map
     */
    GenericNeuron*  GenericHebbianTable::predict_simple()
    {
        int x, y;
        float min = 0;
        for (int i = 0; i < outputSOM->params->som_size; i++)
        {
            for (int j = 0; j < outputSOM->params->som_size; j++)
            {
                if (outputSOM->neurons[i][j]->propagated_activation > min)
                {
                    min = outputSOM->neurons[i][j]->propagated_activation;
                    x = i;
                    y = j;
                }
            }
        }

        // select a random node, if all the activations are 0
        if (min < FLT_EPSILON)
        {
            int i = Utils::random(0,outputSOM->params->som_size-1);
            int j = Utils::random(0,outputSOM->params->som_size-1);
            x = i;
            y = j;
            std::cout<<"random prediction. All activations in output maps are 0. Maybe the hebbian table is not initialise or not connected to any map."<<std::endl;
        }
        return outputSOM->neurons[x][y];
    }

    /**
     * @brief GenericHebbianTable::predict_weightedAverage
     * Predicts with computing a weighted average of the weights in the output map, weighted by the corresponding activations
     * You should propagate at least once from any map (also from other hebbian tables) to the output map, for getting a non-random PREDICTION_TYPE_SIMPLE
     * @return a pointer to a neuron with the weights set to the predicted pattern.
     * The pointer does not point to any actual neuron in the output map
     */
    GenericNeuron*  GenericHebbianTable::predict_weightedAverage()
    {
        GenericNeuron* predictedNode = new GenericNeuron(-1, -1, outputSOM->params);
        //initialise with zeros
        for (int w = 0; w < outputSOM->params->weights_size; w++)
        {
            predictedNode->weights[w] = 0;
        }

        // compute total propagation (since it is not in the range [0,1]
        float total_propagation = 0;
        for (int i = 0; i < outputSOM->params->som_size; i++)
        {
            for (int j = 0; j < outputSOM->params->som_size; j++)
            {
                total_propagation += outputSOM->neurons[i][j]->propagated_activation;
            }
        }
        if (total_propagation < FLT_EPSILON)
        {
            //std::cerr <<"GenericHebbianTable::predict_weightedAverage() - (total_propagation in output map == 0)"<<std::endl;
            //return NULL;
            int i = Utils::random(0,outputSOM->params->som_size-1);
            int j = Utils::random(0,outputSOM->params->som_size-1);
            predictedNode = outputSOM->neurons[i][j];
            //number_of_random_predictions.back() = number_of_random_predictions.back()+1;
            return predictedNode;
        }        


        // weighted average
        for (int i = 0; i < outputSOM->params->som_size; i++)
        {
            for (int j = 0; j < outputSOM->params->som_size; j++)
            {
                for (int w = 0; w < outputSOM->params->weights_size; w++)
                {
                    predictedNode->weights[w] += outputSOM->neurons[i][j]->weights[w] * (outputSOM->neurons[i][j]->propagated_activation / total_propagation);
                }
            }
        }

        return predictedNode;
    }


    // simple prediction takes the winner node in inputSOM and selects the best matched node in the outputSOM
    // according to the function simplePredict
    /**
     * @brief GenericHebbianTable::computePredictionError
     * Compute the prediction error to an observed output.
     * Be careful, as the function does an initial clearing of the propagations in the output map,
     * then propagates the input specified by input_patter and, thus, computes the distance to the prediction
     * @param observed_output the observed output pattern in the output map
     * @param input_pattern the input patter
     * @return the distance between the prediction in the output map and the observed output
     */
    float GenericHebbianTable::computePredictionError(std::vector<float> *observed_output, std::vector<float> *input_pattern)
    {
        if ((observed_output->size()!=outputSOM->params->weights_size) ||(input_pattern->size()!=inputSOM->params->weights_size))
        {
            std::cerr<<"GenericHebbianTable:computePredictionError. Wrong size of input patterns"<<std::endl;
            std::cerr<<"observed_output->size() " <<observed_output->size()<<std::endl;
            std::cerr<<"outputSOM->params->weights_size " <<outputSOM->params->weights_size<<std::endl;
            std::cerr<<"input_pattern->size() " <<input_pattern->size()<<std::endl;
            std::cerr<<"inputSOM->params->weights_size " <<inputSOM->params->weights_size<<std::endl;
        }

        std::vector<float> normalised_observed_output = outputSOM->normalise_input(observed_output);
        std::vector<float> normalised_input = inputSOM->normalise_input(input_pattern);


        outputSOM->clearIncomingActivations();
        propagateInputToOutputMap(&normalised_input);
        return outputSOM->getDistance(&normalised_observed_output, &predict()->weights);
    }

    /**
     * @brief GenericHebbianTable::getPredictionError
     * Get the distance between the output pattern in the output map and the prediction in the output map
     * It requires that you propagated at least once from any hebbian table to the output map
     * @param output_pattern the output pattern in the output map to compare to the prediction
     * @return the distance between output_pattern and the prediction
     */
    float GenericHebbianTable::getPredictionError(std::vector<float> *output_pattern)
    {
        if (output_pattern->size() != outputSOM->params->weights_size)
        {
            std::cerr<<"GenericHebbianTable::getPredictionError(std::vector<float> *output_pattern) : (output_pattern->size() != outputSOM->params->weights_size)"<<std::endl;
            return -1;
        }
        std::vector<float> normalised_observed_output = outputSOM->normalise_input(output_pattern);
        return outputSOM->getDistance(&normalised_observed_output, &predict()->weights);
    }

    /**
     * @brief GenericHebbianTable::normaliseWeights
     * For each node in the input map, it normalises all the connections from it to the output make_pair
     * so that the weights of these links sum up to one. This in fact implements a forgetting process,
     * as non updated links become weaker over time.
     * This has been inspired on the method proposed by Miikkulainen
     * "A Distributed Artificial Neural Network Model Of Script Processing And Memory"
     */
    void GenericHebbianTable::normaliseWeights()
    {
        for (unsigned int x1 = 0; x1<inputSOM->params->som_size; x1++)
        {
            for (unsigned int y1 = 0; y1<inputSOM->params->som_size; y1++)
            {
                float sum = 0;
                std::vector<int> xx2;
                std::vector<int> yy2;
                for (unsigned int x2 = 0; x2<outputSOM->params->som_size; x2++)
                {
                    for (unsigned int y2 = 0; y2<outputSOM->params->som_size; y2++)
                    {
                        if (axons[x1][y1][x2][y2] > 0)
                        {
                            // save the indexes, for faster computation of the normalisation
                            // since the axons matrix is sparse, in case of learnWithWinners and with winners/neighbours
                            xx2.push_back(x2);
                            yy2.push_back(y2);
                            //sum += pow(axons[x1][y1][x2][y2],2);
                            sum += axons[x1][y1][x2][y2];
                        }
                    }
                }

                for (unsigned int i=0; i< xx2.size(); i++)
                {
                    if (sum!=0)
                        axons[ x1 ][ y1 ][ xx2[i] ][ yy2[i] ] = axons [ x1 ][ y1 ][ xx2[i] ][ yy2[i] ] / (sum);
//                    if (axons[ xx1[i] ][ yy1[i] ][ x2 ][ y2 ] < 0.0001)
//                        axons[ xx1[i] ][ yy1[i] ][ x2 ][ y2 ]  = 0;
                }
            }
        }
        std::cout<<"normalised"<<std::endl;
    }

    /**
     * @brief GenericHebbianTable::loadTable
     * Load the hebbian table from a file. Do not forget to run connectSOMs to link the input and
     * output maps, after loading the table.
     * @param filename The name of the file (include the path)
     */
    void GenericHebbianTable::loadTable(std::string filename)
    {
        std::cout<<"Loading hebbian table from "<<filename.c_str()<<std::endl;
        std::ifstream inputfile;
        inputfile.open(filename.c_str(), std::ifstream::in);
        if (!inputfile.is_open())
        {
            std::cerr<<"Could not open "<<filename.c_str()<<"\n";
            return;
        }

        std::string line;
        getline (inputfile, line);
        getline (inputfile, line);

        char c;
        int val_int;
        float val_float;
        std::istringstream header_iss(line);
        header_iss>>c; // skip #
        header_iss>>inputSOM->params->som_size;
        header_iss>>outputSOM->params->som_size;
        header_iss>>param->hebbian_learning_rate;
        header_iss>> param->hebbian_rule_type;
        header_iss>> param->learning_strategy;
        int norm_type;
        header_iss>> norm_type;
        inputSOM->params->normalisation_type = norm_type;
        outputSOM->params->normalisation_type = norm_type;
        header_iss>> param->neighbours_size;
        header_iss>> param->buffer_size;

        axons.resize(inputSOM->params->som_size);
        for (unsigned int i=0; i<inputSOM->params->som_size; i++)
        {
            axons[i].resize(inputSOM->params->som_size);
            for (unsigned int j=0; j<inputSOM->params->som_size; j++)
            {
                axons[i][j].resize(outputSOM->params->som_size);
                for (unsigned int k=0; k<outputSOM->params->som_size; k++)
                {
                    axons[i][j][k].resize(outputSOM->params->som_size);
                    for (unsigned int q=0; q<outputSOM->params->som_size; q++)
                    {
                      axons[i][j][k][q]=0;
                    }
                }
            }
        }

        int x1, y1, x2, y2;

        while (getline (inputfile, line))
        {
            std::istringstream iss(line);
            iss>>x1;
            iss>>y1;
            iss>>x2;
            iss>>y2;
            iss>>axons[x1][y1][x2][y2];
        }

        std::cout<<"Hebbian table loaded."<<std::endl;
    }



    /**
     * @brief GenericHebbianTable::saveTable
     * save the table into a file with the specified filename
     * @param filename include the path
     */
    void GenericHebbianTable::saveTable(std::string filename)
    {
        std::ofstream myfile;
        myfile.open (filename.c_str());
        myfile<<"# inputSOM->params->som_size outputSOM->params->som_size hebbian_rule_type learning_rate normalisation_type neighbour_size buffer_size"<<std::endl;
        myfile<<"# "<<inputSOM->params->som_size<<" "<< outputSOM->params->som_size<<" "<< param->hebbian_learning_rate<<" "<<param->hebbian_rule_type<<" "<< param->learning_strategy<< " "<< inputSOM->params->normalisation_type<<" "<<param->neighbours_size<<" "<<param->buffer_size<<std::endl;
        for (int x1=0; x1<inputSOM->params->som_size; x1++)
        {
            for (int y1=0; y1<inputSOM->params->som_size; y1++)
            {
                for (int x2=0; x2<outputSOM->params->som_size; x2++)
                {
                    for (int y2=0; y2<outputSOM->params->som_size; y2++)
                    {
                        myfile<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<axons[x1][y1][x2][y2]<<std::endl;
                    }
                }
            }
        }
       myfile.close();

    }


    /**
     * @brief GenericHebbianTable::printLinks
     * Print all the connections stored in the hebbian table
     */
    void GenericHebbianTable::printLinks()
    {
        std::cout<<"Hebbian Links"<<std::endl;
        for (int x1=0; x1<inputSOM->params->som_size; x1++)
        {
            for (int y1=0; y1<inputSOM->params->som_size; y1++)
            {
                for (int x2=0; x2<outputSOM->params->som_size; x2++)
                {
                    for (int y2=0; y2<outputSOM->params->som_size; y2++)
                    {
                        std::cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<axons[x1][y1][x2][y2]<<std::endl;
                    }
                }
            }
        }
    }


    /**
     * @brief GenericHebbianTable::getTable
     * Returns the connections stored in the hebbian table
     */
    std::vector<float>  GenericHebbianTable::getTable()
    {
        std::vector<float> output;
        for (int x1=0; x1<inputSOM->params->som_size; x1++)
        {
            for (int y1=0; y1<inputSOM->params->som_size; y1++)
            {
                for (int x2=0; x2<outputSOM->params->som_size; x2++)
                {
                    for (int y2=0; y2<outputSOM->params->som_size; y2++)
                    {
                        output.push_back(axons[x1][y1][x2][y2]);
                    }
                }
            }
        }
        return output;
    }

}


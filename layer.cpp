/*
* 	Library Module Implementing Layers of Perceptrons for CY2D7 .. SE2NN11
*
* 	Adapted by Alexander Wolff	20/10/14
*/

#ifndef LAYER_CPP
#define LAYER_CPP

#include "Header/library.h"


///<summary>
/// Returns a random number of type double, between -1 and 1
///</summary>
double myrand () 
{			
	// return a random number in the range -1..1
	// do so calling the rand function in math library
   return -1.0 + (2.0 * rand() / RAND_MAX);
}

// Implementation of LinearLayerNetwork *****************************

///<summary>
/// Initialises variables, and allocates space for arrays
/// There are as many neurons in the layer as the number of outputs
/// All neurons in the layer share the same inputs
///
///<argument="int numIns">Number of inputs</argument>
///<argument="int numOuts">Number of outputs</argument>
///</summary>
LinearLayerNetwork::LinearLayerNetwork (int numIns, int numOuts) { 
	 
	//Store number of inputs
	numInputs = numIns;							
	
	//Store number of outputs
	numNeurons = numOuts;						
	
	//Calculate and store number of weights
	// "+ 1" refers to the bias
	numWeights = (numInputs + 1) * numNeurons;	
				
					
	//Allocate space for the output array							
	outputs = new double [numNeurons];			
	
	//Allocate space for the delta array	
	deltas = new double [numNeurons];			
    
    //Allocate space for the weight array	
    weights = new double [numWeights];			
    
    //Allocate space for the delta-weight array	
    deltaWeights = new double [numWeights];	
    
    	

	for (int i=0; i < numWeights; i++)  
	{
		//Initialise weights to random value between -1 and 1
		weights[i] = myrand();					
		
		//Initialise delta-weights to 0
		deltaWeights[i] = 0;
    }
    
	for (int i=0; i < numNeurons; i++) 
	{
		//Initialise outputs to 0
		outputs[i] = 0;
		
		//Initialise deltas to 0
		deltas[i] = 0;
	}
}


///<summary>
/// Destructor for the Object: LinearLayerNetwork, frees memory
///</summary>
LinearLayerNetwork::~LinearLayerNetwork() {

	delete [] weights; 					
    delete [] deltaWeights;				
	delete [] outputs;					
	delete [] deltas; 					
}


///<summary>
/// Calculates and stores the output of each neuron in the layer
/// Equation (for one neuron):
/// OUTPUT = SUM{ INPUT * WEIGHT }
///
///<argument="const double inputs[]">Array countaining the inputs</argumment>
///</summary>
void LinearLayerNetwork::CalcOutputs(const double inputs[]) {

	//Tracks which weight is being accessed
	int weight_index = 0;
	
	for (int neuron_counter=0; neuron_counter < numNeurons; neuron_counter++) 
	{	
		//Processes each neuron in order
		outputs[neuron_counter] = weights[weight_index++];
		
		//The summation is done here
		for (int input_counter=0; input_counter < numInputs; input_counter++)
			outputs[neuron_counter] += inputs[input_counter] * weights[weight_index++];
	}
}


///<summary>
/// Passes each item in the dataset to the network then calculates and stores the outputs
///
///<argument="dataset &data">Location to the dataset containing data to be tested</argument>
///</summary>
void LinearLayerNetwork::ComputeNetwork (dataset &data) {

	//For each item in the data-set
	for (int i=0; i < data.numData(); i++) 
	{ 
		//Calculates outputs
	    CalcOutputs (data.GetNthInputs(i));
	    
	    //Save outputs into data-set
		StoreOutputs(i, data);
	}
}


///<summary>
/// Stores the calculated array of outputs inside a dedicated array inside the dataset.
///
///<argument="int n">Space in the 2D array in which to store 1D array of outputs</argument>
///<argument="dataset &data">Location to the dataset containing data to be tested</argument>
///</summary>
void LinearLayerNetwork::StoreOutputs (int n, dataset &data) {

	//Passes output-array into the dataset
	data.SetNthOutputs(n, outputs);
}


///<summary>
/// Finds and stores the deltas from the errors. It is assumed the size of errors-array and the size of deltas-array are equal.
/// Equation (for a linear system):
/// delta = Error
///
///<argument="const double errors[]">Array of errors used to find deltas</argument>
///</summary>
void LinearLayerNetwork::FindDeltas (const double errors[]) {
	
	////only copying has to be done, there are as many errors as there are outputs and as many outputs as there are neurons
	dcopy(numNeurons, errors, deltas);
	
}


///<summary>
/// Calculates and stores the new weights from the errors.
/// Equation (for a linear system):
/// new weight = old weight + ( (error * input * learning_rate) + momentum + old change in weight)
///
///<argument="const double inputs[]">Array of inputs used to find the new weights</argument>
///<argument="const double learningParameters[]"> Array containing the parameters: {learning-rate, momentum}</argument>
///</summary>
void LinearLayerNetwork::ChangeAllWeights (const double Inputs[], const double learningParameters[]) {

	//Used to keep track of the current input
	double current_input;
	
	//Used to keep track of which weight is being used
	int weight_index = 0;

	//For each neuron in the layer
	for(int neuron_index=0; neuron_index < numNeurons; neuron_index++)
	{
		
		//For each input in the input-array
		for(int input_index=0; input_index < numInputs + 1; input_index++)
		{
			//IF bias weight
			if((input_index % (numInputs + 1))==0) current_input = 1;
			else current_input = Inputs[input_index - 1];
			
			//Equate  (delta * input * learning_rate)  ADD  (momentum * previous_delta)
			deltaWeights[weight_index] = (current_input * deltas[neuron_index] * learningParameters[0])
											+ (deltaWeights[weight_index] * learningParameters[1]);
			
			//New weight = old weight + change in weight
			weights[weight_index] += deltaWeights[weight_index];
			
			//Move on to the next weight
			weight_index++;
		}	
	}
}



void LinearLayerNetwork::AdaptNetwork (dataset &data, const double learningParameters[]) {
		// pass whole dataset to network : for each item
		//   calculate outputs, copying them back to data
		//   adjust weights using the delta rule : targets are in data
		//     where learnparas[0] is learning rate; learnparas[1] is momentum

	for (int i=0; i<data.numData(); i++) 
	{
			// for each item in data set
		CalcOutputs(data.GetNthInputs(i));
			// get inputs from data and pass to network to calculate the outputs
		StoreOutputs (i, data);
			// return calculated outputs from network back to dataset 
		FindDeltas(data.GetNthErrors(i));
			// get errors from data and so get neuron to calculate the delta
		ChangeAllWeights(data.GetNthInputs(i), learningParameters);
			// and then change all the weights, passing inputs and learning constants
	}
}



void LinearLayerNetwork::SetTheWeights (const double initialWeights[]) {
	// set the weights of the layer to the values in initWeights

	// do so by copying from initialWeights into object's weights
   dcopy (numWeights, initialWeights, weights);	
   // copy all weights (numWeights says how many) from array initialWeight to layer's array weights
}



int LinearLayerNetwork::HowManyWeights (void) {
	// return the number of weights in layer

	return numWeights;
}


///<summary>
///	Copies the array of weights stored in the network-object to the currentWeights array
///
///<argument="double currentWeights[]">Array containing the current weights</argument>
///</summary>
void LinearLayerNetwork::ReturnTheWeights (double currentWeights[]) {
	
	//Copies as many elements as there are weights, from the weights array, to the currentWeights array
	dcopy (numWeights, weights, currentWeights);
} 


///<summary>
/// Calculates and returns the errors of the previous layer, only works if the output only has one neuron
///
///<argument="double previousErrors[]"> Array in which previous errors are to be stored </argument>
///</summary>
void LinearLayerNetwork::PrevLayersErrors (double previousErrors[]) {

	/*
	//CODE for only ONE node in output layer
	//Each previous neuron output is an input to this network
	int previousNeurons = numInputs;
	
	//There is only one neuron in the output layer, therefore only use deltas[0]
	for(int i=0; i < previousNeurons; i++)
	{
		previousErrors[i] = deltas[0] * weights[i+1];
	}
	*/

	
	//For each input 
	for(int i=0; i < numInputs; i++)
	{
		
		previousErrors[i] = 0;
		
		//For each output node
		for(int j=0; j < numNeurons; j++)
		{
			//Accounting for the bias weights and inputs
			previousErrors[i] += deltas[j] * weights[((numInputs+1)*j)+(i+1)];
			
		}
		
	}
	
}

// Implementation of SigmoidalLayerNetwork *****************************

SigmoidalLayerNetwork::SigmoidalLayerNetwork (int numInputs, int numOutputs):LinearLayerNetwork (numInputs, numOutputs) 
{
	// just use inherited constructor - no extra variables to initialise
}

SigmoidalLayerNetwork::~SigmoidalLayerNetwork() 
{
	// destructor - does not need to do anything other than call inherited destructor
}



///<summary>
///	Calculates the outputs of the sigmoidal layer
/// Equation:
/// Temp output = input * weight
/// output = 1 / (1 + exp( - sum ) )
///
///<argument="const double inputs[]">Array containing the inputs</argument>
///</summary>
void SigmoidalLayerNetwork::CalcOutputs(const double inputs[]) {		
	// Calculate outputs being Sigmoid (WeightedSum of ins)
	
	//Makes use of inheritance to find outputs as done with linear-activation networks
	LinearLayerNetwork::CalcOutputs(inputs);
	
	
	for (int neuron_counter=0; neuron_counter < numNeurons; neuron_counter++) 
	{	
		//Actual output = 1 / (1 + exp( - temp_output ) )
		outputs[neuron_counter] = 1 / (1 + exp( -1 * outputs[neuron_counter]));
	} 
}

///<summary>
/// Calculates and stores the deltas for the sigmoidal layer
/// Equation:
/// Deltas = Outputs * (1 - Outputs) * Errors
///
///<argument="const double errors[]">Array containing the errors</argument>
///</summary>
void SigmoidalLayerNetwork::FindDeltas (const double errors[]) {		
	
	for(int output_index=0; output_index < numNeurons; output_index++)
	{
		//Calculates delta for each neuron
		deltas[output_index] = outputs[output_index] * (1 - outputs[output_index]) * errors[output_index];
	}
	
}

/* Implementation of MultiLayerNetwork *****************************/

///<summary>
/// Constructor
///
///<argument="int numInputs"> Amount of inputs to the layer</argument>
///<argument="int numOutputs"> Amount of outputs to the layer</argument>
///<argyment="LinearLayerNetwork *to_next_layer"> Pointer to the next layer</argument>
///</summary>
MultiLayerNetwork::MultiLayerNetwork (int numInputs, int numOutputs, LinearLayerNetwork *to_next_layer) :SigmoidalLayerNetwork (numInputs, numOutputs) 
{
	// Construct a hidden layer with numInputs inputs and numOutputs outputs
	// Where (a pointer to) its next layer is in to_next_layer
	to_next_layer = new SigmoidalLayerNetwork(numInputs, numOutputs);

	// Attach the pointer to the next layer that is passed
	nextlayer = to_next_layer;
}

///<summary>
/// Destructor
///</summary>
MultiLayerNetwork::~MultiLayerNetwork()
{
	// Remove output layer 
	delete nextlayer;
	
	// Automatically-calls inherited destructor
}

///<summary>
/// Calculates outputs from weights and inputs
///
///<argument="const double Inputs[]"> An array containing inputs to the network</argument>
///</summary>
void MultiLayerNetwork::CalcOutputs(const double Inputs[]) 
{
		// Calculate the outputs of the main layer given the Inputs[]
		SigmoidalLayerNetwork::CalcOutputs(Inputs);
		
		// Calculates the outputs of the next layer using the outputs of the main layer as input
		nextlayer->CalcOutputs(outputs);

}

///<summary>
/// Copies the calculated network outputs into the nth output in data.
///
///<argument="int n"> Directs where to store outputs</argument>
///<argument="dataset &data"> Pointer to the dataset</argument>
///</summary>
void MultiLayerNetwork::StoreOutputs(int n, dataset &data) 
{
		// Store calculated network's outputs into the nth output set in data
		nextlayer->StoreOutputs(n, data);
}

///<summary>
/// Calculates the deltas in this layer and the next layer
///
///<argument="const double Errors[]"> Array containing the errors of the network (target - output)</argument>
///</summary>
void MultiLayerNetwork::FindDeltas (const double Errors[]) 
{	
	//Find deltas and errors in the next layer
	nextlayer->FindDeltas(Errors);
	
	//Prepare an array to contain the errors
	double thisErrors[numNeurons];
	
	//Loads the previous errors
	nextlayer->PrevLayersErrors(thisErrors);
	
	//Convert to deltas for sigmoidal activation
	SigmoidalLayerNetwork::FindDeltas(thisErrors);
}

///<summary>
/// Changes all the weights in this layer and the next 
///
///<argument="const doublt Inputs[]"> Array containing the inputs to the layer</argument>
///<argument="const double learningParameters[]"> Array containing the parameters: {learning-rate, momentum}</argument>
///</summary>
void MultiLayerNetwork::ChangeAllWeights (const double Inputs[], const double learningParameters[]) 
{	
	//Change weights for this layer
	SigmoidalLayerNetwork::ChangeAllWeights(Inputs, learningParameters);
	
	//Change weights for the next layer
	nextlayer->ChangeAllWeights(outputs, learningParameters);
}

///<summary>
/// Initialises the weights in the main layer and next layers in the network using the values in initialWeights[]
///
///<argument="const double initialWeights[]"> Array containing initial weights
/// contains weights for the main layer and next layer[0, ..., n, ..., /0] </argument>
///</summary>
void MultiLayerNetwork::SetTheWeights (const double initialWeights[])
{
	//Makes explaination easier
	int n = numWeights;

	//Weights 0 to n
	LinearLayerNetwork::SetTheWeights(initialWeights);
	
	//Weights n to /0 (end)
	nextlayer -> SetTheWeights(&initialWeights[n]);
}


///<summary>
/// Returns the amount of weights in this layer and the next as an integer
///</summary>
int MultiLayerNetwork::HowManyWeights () 
{
	return (numWeights + nextlayer->numWeights);
}

///<summary>
/// Copies the weights of main layer and next layer into the argument array: "theWeights[]"
///
///<argument="double theWeights[]"> Array onto the network's weights are copied</argument>
///</summary>
void MultiLayerNetwork::ReturnTheWeights (double theWeights[]) 
{
	//Contains all the weights of the network
	double buffer [HowManyWeights()];
	
	//Stores weights from current layer
	for(int i=0; i<numWeights; i++)
	{
		buffer[i] = weights[i];
	}
	
	//Stores weights from the next layer
	for(int i=0; i<(nextlayer->numWeights); i++)
	{
		buffer[i+numWeights] = nextlayer->weights[i];
	}
	
	//Copies the buffer array into the output array
	dcopy (HowManyWeights(), buffer, theWeights);
}

#endif

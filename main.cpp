/*
* 	SIMPLE MULTI-LAYER PERCEPTRON PROGRAM for SE2NN11
*
* 	Has single/multi layer network with multiple inputs and outputs.
* 	Can have linear or sigmoidal activation.
* 	Configuration is set by files containing data sets used.
* 	Can be tested on simple logic problems or numerical problems.
* 
* 	Adapted by Alexander Wolff	20/10/14
*/

///<summary>
///	The library contains all customly-designed classes: 
///	{layer.h, layer.cpp} for implementation of single and multiple layers of perceptrons
///	{data.h, data.cpp} for input/output manipulation and storing input/output/threshold/error
///
///	The library also links to the necessary stock libraries such as:
/// {iomanip, fstream, iostream, math.h, stdlib.h, string.h}
///</summary>
#include "Header/library.h"



///<summary>
/// Tests the network by passing each element in dataset: data to network: net
///
///<argument="LinearLayerNetwork *net"> Pointer linear network of perceptrons</argument>
///<argument="dataset &data"> Location to the dataset containing data to be tested</argument>
///<argument="int option"> Changed mode of the function:
///	IF = 0: prints SSE and % classifications
///	IF = 1: prints above + inputs and outputs
/// IF = -1: doesn't print anything</argument>
///</summary>
void TestTheNet (LinearLayerNetwork *net, dataset &data, int option) {

	//Passes dataset to network	
	net -> ComputeNetwork (data);
	
	//Printing mode as described in summary
	if (option >= 0) data.printdata(option);
}				


///<summary>
/// Initialises the network's weights, the weights can be predefined or randomised
///
/// <argument="LinearLayerNetwork *net"> Network to have weights initialised</argument>
/// <argument="int option"> Mode at which the function opperated
/// IF = 'X': weights initialised with Picton's weights, given from his book
/// IF = 'O': weights initialised for non-linear setup
/// IF = 'L' or 'S':  weights initialised with weights provided in R.J.Mitchell's lectures
/// ELSE: weights are randomised, this is the default mode</argument>
///</summary>
void SetTheWeights (LinearLayerNetwork *net, char option) {

	//weights given in Picton's book
	double picton_weights[] = {0.862518, -0.155797,  0.282885,
							  0.834986, -0.505997, -0.864449,
							  0.036498, -0.430437,  0.481210};
	
	// initial weights for neuron layer as used in lectures
	double lecture_weights[] = {0.2, 0.5, 0.3, 
								0.3, 0.5, 0.1, 
								0.4, 0.1, 0.2};	
				
	// initial weights for non-lin setup	
	double nonlinear_weights[] = {	0.2, 	0.5,	0.4,	0.1, 	
									0.2,	0.4,	-0.1, 	0.5,	
									0.7, 	-0.2, 	-0.7,	0, 	
									0,  	0.1, 	-0.3, 	0.3, 	
									0.4, 	0.1,	0.1, 	0.2, 	
									0.3, 	0.2, 	0.1, 	0.2,
									-0.2, 	0.4, 	-0.5, 	0.1};
	                    
	switch(option)
	{	
		case 'X': //IF doing XOR, use this mode
		net -> SetTheWeights (picton_weights); break;
		
		case 'O': 
		net -> SetTheWeights (nonlinear_weights); break;
		
		case 'L': //IF doing logic problem, use this mode
		case 'S': 
		net -> SetTheWeights (lecture_weights); break; 
		
		default: 
		break;
	}
}


///<summary>
/// Creates and returns a neural-layer
///
///<argument="char option"> Controls the mode of operation of the function:
/// IF = 'L': Creates and returns linear activation layer
/// IF = 'S': Creates and returns sigmoidal activation layer 
/// ELSE: Creates and returns multi-layer sigmoidal activation network</argument>
///<argument="int hiddenNodes">Number of nodes in the hidden layer</argument>
///<argument="dataset &data">Location to the dataset containing data to be used</argument>
///
///<return="LinearLayerNetwork*">Layer of neurons</return>
///</summary>
LinearLayerNetwork * MakeNet (char option, int hiddenNodes, dataset &data) {
								
	switch(option)
	{
		case 'L': //Creates and returns single Linear activation layer
		return new LinearLayerNetwork (data.numIns(), data.numOuts());
		
		case 'S': //Creates and returns single Sigmoidal activation layer
		return new SigmoidalLayerNetwork (data.numIns(), data.numOuts());
		
		default: //Creates and returns Milti Sigmoidal activation layer
		return new MultiLayerNetwork (data.numIns(), hiddenNodes, new SigmoidalLayerNetwork (hiddenNodes, data.numOuts()) );
		
	}
}

						
///<summary>
/// Gets character input from user, sets letter to upper case if it isn't so already.
///
///<return="char input">User upper case of inputed character</return>
///</summary>
char getcapch(void) {

	char input;
	
	//Get character
	cin >> input;
	
	//Ignores rest of the line						
	cin.ignore(1);

	//IF input is lower case
	if ( (input >= 'a') && (input <= 'z')) 
	{
		//Convert to upper-case
		input = input - 32;
	}
	
	return input;
}


///<summary>
/// Prints wrights of the neurons in the given network to console
///
///<argument="LinearLayerNetwork *net">Pointer to a neural-network</argument>
///</summary>
void showweights (LinearLayerNetwork *net) {

	// Allocate sufficient memory to store all the weight values
	double *weights = new double [net->HowManyWeights()];
			
	//Copy array of weights into the created weights-array		
	net->ReturnTheWeights (weights);

			
	for (int i = 0; i < ( net->HowManyWeights() ); i++)
	{
		//Print the weights
		cout << weights[i] << ',';
	}
	
	//Flush the buffer
	cout << endl;
	
	//Free the memory taken up by the weights-array
	delete weights;
}


///<summary>
/// Requests user to set the learning rate and momentum
///
///<argument="double* learningParameters"> Array containing the parameters: {learning-rate, momentum}</argument>
///</summary>
void setlparas (double* learningParameters) {
			
	cout << "Enter Learning Rate: [range 0 to 1]\t" << flush;
	
	cin >> learningParameters[0];
	
	cout << "Enter Momentum: [range 0 to 1]\t" << flush;
	
	cin >> learningParameters[1];
	
	//Enforces boundaries
	for(int i=0; i<2; i++)
	{
		if(learningParameters[i] < 0) learningParameters[i] = 0;
		if(learningParameters[i] > 1) learningParameters[i] = 1;
	}
}


///<summary>
/// This function is a sub-menu from which functions relating to the initialisation
/// and learning of the network.
/// Allows the user to see what the network is outputting as well as its weights
/// Allows the user to change learning parameters
///
///<argument="char network_option"> Controls the function's operating mode:
/// IF = 'L': network is linear, so set maximum amount of Epochs to 7
/// ELSE: set maximum amount of Epochs to 1001</argument>
///<argument="int weight_option"> Controls which weights are to be used:
/// IF = 0: sets specific weights
/// ELSE: sets random weights</argument>
///<argument="int hiddenNodes"> Number of hidden nodes to be used in a multi-layered network</argument>
///<argument="char* filename"> Path to file from which to load data</argument>
///<argument="char* dataname"> Name to be given to the data</argument>
///<argument="double* learningParameters">Array containing the parameters: {learning-rate, momentum}</argument>
///</summary>
void testnet (char network_option, int weight_option, int hiddenNodes, char* filename, char* dataname, double* learningParameters) {
	
	//Maximum amount of Epochs
	int max_epoch;
	
	//Epoch number so far
	int current_epoch = 0;
	
	//Used in the menu-selection
	char user_input = 0;
	
	//Loads data from file
	dataset data (filename, dataname);			

	//If the file cannot be loaded
	if (data.numIns() == 0)  
	{
		cout << dataname << " [!] File not found : May be in wrong directory" << endl;
		
		//Abort function
		return;
	}

	//Initialise random-number generator
	srand(weight_option);
	
	//IF statement is true:
	//For linear-activation networks
	if (network_option == 'L')
	{ 
		//For linear-activation networks
		max_epoch = 7;
	} else {
	
		//For sigmoidal-activation networks
		max_epoch = 1001;
	}
	
	//Creates the appropriate neural-network
	LinearLayerNetwork *net = MakeNet (network_option, hiddenNodes, data);
	
	
	//IF statement is true:
	//For when weights are not random						
	if (weight_option == 0) 
	{
		//Sets weights as appropriate
		SetTheWeights (net, network_option);
	}

	
	//IF not of type:Classifier, tests the untrained network and prints results to the console
	TestTheNet (net, data, 1);		

	//Flag for when the sub-menu needs to be exited
	bool exit = false;
	
	//Selection menu
	while(!exit)
	{
		cout << endl << "SELECT:" << endl 
					 << "[L]earn. [P]resent Data. [C]hange Learning Constants. Find [W]eights. [S]ave Learnt Data. [A]bort." << endl
					 << ">" << flush;
		user_input = getcapch(); 
		
		switch(user_input)
		{
			case 'A': //Mode: Abort
			exit = true; break;
			
			case 'L': //Mode: Learn
			{
				for(int i = 0; i < max_epoch; i++)
				{
					//Passes data to the network and updates weights
					net -> AdaptNetwork (data, learningParameters);
					
					//IF statement is true:
					//Every Epochs for linear-activation networks
					//Every 200 Epochs for sigmoidal-activation networks
					if( (max_epoch < 10) || ((i % 200) == 0))
					{
						//Prints the Epoch number 
						cout << "\n\tEpoch " << setw(6) << i + current_epoch << endl;
		
						//Prints the relevant data
						data.printdata(0);
					}
					
				}
				//Updates current Epoch to its true value
				current_epoch = current_epoch + max_epoch - 1;
			} break;
			
			case 'C'://Mode: Change Learning Constants
			setlparas(learningParameters); break;
			
			case 'P'://Mode: Present Data
			TestTheNet (net, data, 1); break;
			
			case 'W'://Mode: Find Weights
			showweights (net); break;
			
			case 'S'://Mode: Save Learnt Data
			{
				//Pass data to the network
				TestTheNet (net, data, -1);
				
				//Save the data to file
				data.savedata();			
			} break;
			
			//Ignore unrecognised inputs
			default: break;
		}
	}
}


///<summary>
/// Exposes a network against a training set and an unseen set, saves the end result 
/// so that it can be plotted using tadpole
///
///<argument="double* learningParameters">Array containing the parameters: {learning-rate, momentum}</argument>
///<argument="int hiddenNeurons">Number of hidden neurons to be used in the network</argument>
///<argument="int max_epochs">Maximum amount of Epochs</argument>
///<argument="int weight_option">Controls which weights are to be used:
/// IF = 0: sets specific weights
/// ELSE: sets random weights</argument>
///<argument="char *training_set">Used to create the training set</argument>
///<argument="char *unseen_set">Used to create the unseen set</argument>
///</summary>
void classtest (double* learningParameters, int hiddenNeurons, int max_epoch, int weight_option, char *training_set, char *unseen_set) {
	
	//Unused variables?
	//(previous sum of valid.SSE, current sum)
	//double vwas = 1000, vsum = 0;			
	
	//Initialises random-number generator
	srand(weight_option);
	
	//Creates the training set
    dataset train (training_set, "iristrain");
    
    //Creates the unseen set
	dataset unseen (unseen_set, "irisunseen");
	
	//Creates a network with given nymber of hidden neurons
	LinearLayerNetwork * net = MakeNet ('N', hiddenNeurons, train);
	
	//Test the training set on the untrained network
	TestTheNet (net, train, 0);				
	
	//Test the unseen set on the network
	TestTheNet (net, unseen, 0);			
	
	//Epoch number so far
	int current_epoch; 

	//Train until max_epoch is reached
	for( current_epoch = 0; current_epoch < max_epoch; current_epoch++ )
	{
		//Pass the training set to the network
		net -> AdaptNetwork (train, learningParameters);
		
		//IF statement is true:
		//Every 50 epochs
		if( (current_epoch % 50) == 0)
		{
			//Prints current Epoch
			cout << "\t" << current_epoch << flush;
			
			//Prints relevant data
			train.printarray(" ", 'C', 0, -1);
		}
	}
	
	//Tests the training set on the trained network AND prints %(percentage) classification
	TestTheNet (net, train, 0);
	
	//Saves data in a file tadpole can use to plot
	train.savedata(1);
	
	//Tests the unseen set on the trained network AND prints %(percentage) classification
	TestTheNet (net, unseen, 0);
	
	//Saves data in a file tadpole can use to plot
	unseen.savedata(1);						
}


///<summary>
/// Tests the network against a numerical problem using the specified parameters
///
///<argument="double *learningParameters">Array containing the parameters: {learning-rate, momentum}</argument>
///<argument="int hiddenNeurons">Number of hidden neurons in the network</argument>
///<argument="int max_epoch">Maximum amount of epochs</argument>
///<argument="int usevalid">If usevailid is TRUE, then stop training when SEE on validation set starts to rise</argument>
///<argument="int weight_option">Controls which weights are to be used:
/// IF = 0: sets specific weights
/// ELSE: sets random weights</argument>
///<argument="char *training_set">Path and filename for the training set</argument>
///<argument="char *validation_set">Path and filename for the validation set</argument>
///<argument="char *unseen_set">Path and filename for the unseen set</argument>
///</summary>
void numtest (double* learningParameters, int hiddenNeurons, int max_epoch, int usevalid, int weight_option, char *training_set, char *validation_set, char *unseen_set) 
{
	
	//Initialise Random-Number Generator
	srand(weight_option);
	
	//Declare and Initialise dataset for the Training Set
	dataset train (training_set, "Training_set");
	
	
		//Declare and Initialise dataset for the Validation Set
		dataset validation (validation_set, "Validation_set");
	
	
	//Declare and Initialise dataset for the Unseen Set
	dataset unseen (unseen_set, "Unseen_set");
	
	//Create network (using MakeNet() )
	LinearLayerNetwork *net = MakeNet (usevalid, hiddenNeurons, train);
	
	//Show Training Set to untrained network and report SSE
	TestTheNet (net, train, 0);
	
	//IF usevalid is TRUE, show Validation Set to untrained network and report SSE 
	if(usevalid) TestTheNet(net, validation, 0);
	
	//Show Unseen set to trained network and report SSE
	TestTheNet (net, unseen, 0);
	
	
	//Epoch sentinel
	int current_epoch;
	
	
		//Average Sum of Squared Errors in 10 epochs: in the validation-set: for the current iteration
		double current_average_SSE = 0;
		
		//Average Sum of Squared Errors in 10 epochs: in the validation-set: for the previous iteration
		double previous_average_SSE = 0;

	
	//FOR each epoch 
	for(current_epoch = 0; current_epoch < max_epoch; current_epoch++)
	{
	
		//Pass training set to network
		net -> AdaptNetwork (train, learningParameters);
		
		if(usevalid)
		{	
			//Calculates SSE for validation set
			current_average_SSE += validation.TotalSSE();
		}
	
		//Print SSE on training set every 20 epoch
		if( (current_epoch % 20) == 0)
		{
			//Prints current Epoch
			cout << "\t" << current_epoch << flush;
			
			//Prints relevant data
			train.printarray(" ", 'C', 0, -1);
			
		}
		
		//IF usevalid is true: THEN test every 10 Epochs
		if( (usevalid) && ((current_epoch % 10) == 0) )
		{
			//Calculate the true average
			current_average_SSE /= 10;
			
			//Skips first time
			if(previous_average_SSE == 0){}

			//IF at least 150 Epoch have passed THEN test if network should stop learning
			else if( (current_average_SSE > (previous_average_SSE * 0.999)) && (current_epoch > 150) ) 
			{
				break;
			}
			
			printf("\tCurrentSSE[%0.30f]\tPreviousSSE[%0.30f]\n\n", current_average_SSE, previous_average_SSE);
				
			
			//Prepares for next iteration
			previous_average_SSE = current_average_SSE;
			current_average_SSE = 0;
			
		}
	}
	
	//Output number of epochs taken
	printf("\nNumber of Epochs taken: [%d]\n", current_epoch);
	
	//Pass Training Set to trained network and report SSE
	TestTheNet (net, train, 0);
	
	//IF usevalid is TRUE, show Validation Set to trained network and report SSE 
	if(usevalid) TestTheNet(net, validation, 0);
	
	//Pass Unseen set to trained network and report SSE
	TestTheNet (net, unseen, 0);
	
	//Saves data and sets it up for the tadpole plotting program
	unseen.savedata(1);
	
}




///<summary>
/// GUI for the program
///</summary>
int main() {
	
	int weight_option =0;

	int hiddenNeurons = 10;

	int max_epoch = 1001;

	double learningParameters[] = { 0.2, 0.0 };
	
	char user_input = 0;
	
	char network_option = 'L';
	
	char usevalid = 'Y';
	

	cout << "Richard J. Mitchell's Perceptron Network Program\n\tAdapted by Abdelrahmane Bray [Autumn 2014]" << endl;

	
	//Exit-flag for the menu
	bool exit = false;
	
	while(!exit)
	{
		
		cout << endl << "######################" << endl;
		
		cout << endl << endl << "Network is: " << flush;
		
		switch(network_option)
		{
			case 'L':
			cout << "for Linear-activation" << endl; break;
			
			case 'S':
			cout << "for Sigmoidal-activation" << endl; break;
			
			case 'X':
			cout << "for XOR" << endl; break;
			
			case 'O':
			cout << "for Other Non-linear problems" << endl; break;
			
			default:
			cout << "for Linear problems with [" << hiddenNeurons << "] hidden neurons." << endl; break;
		}
		
		cout << "Initial weights seed [" << weight_option << "] " << endl;
		
		cout << "Learning rate: [" << learningParameters[0] << "]. Momentum: [" << learningParameters[1] << "]" << endl;

		cout << endl << "MENU:: Select one of the following:" << endl
			 << "[T]est Network. Set [N]etwork. Set Learning-[C]onstants. [I]nitialise Random Seed. [Q]uit" << endl
			 << ">" << flush;
		
		//Read user input
		user_input = getcapch();
		
		
		switch(user_input)
		{
			case 'Q'://Choice: Quit
			exit = true; break;
			
			case 'T'://Choice: Test Network
			{
				switch(network_option)
				{
					case 'L'://Test: Single layer network 
					case 'S':
					testnet (network_option, weight_option, 4, "Resource/logdata.txt", "AndOrXor", learningParameters); break;
					
					case 'X'://Test against: XOR
					testnet (network_option, weight_option, 2, "Resource/xordata.txt", "XOR", learningParameters); break;
					
					case 'O'://Test against: non-linear problem
					testnet (network_option, weight_option, 4, "Resource/nonlinsep.txt", "NonLinSep", learningParameters); break;
					
					case 'C':
					classtest (learningParameters, hiddenNeurons, max_epoch, weight_option, "Resource/iristrain.txt", "Resource/irisunseen.txt"); break;
					
					case 'U':
					testnet (network_option, weight_option, 4, "Resource/username.txt", "Username", learningParameters); break;
					
					case 'M':
					numtest (learningParameters, hiddenNeurons, max_epoch, usevalid == 'Y', weight_option, "Resource/trainNorm.txt", "Resource/validNorm.txt", "Resource/unseenNorm.txt"); break;
					
					default://Test against: Numerical Problem
					numtest (learningParameters, hiddenNeurons, max_epoch, usevalid == 'Y', weight_option, "Resource/train.txt", "Resource/valid.txt", "Resource/unseen.txt"); break;
				}
				
			}break;
			
			case 'N'://Choice: Set Network
			{
				cout << "SELECT NETWORK:" << endl
					 << "[L]inear. [S]igmoidal. [X]OR. [O]ther non-Separable. [C]lassifier. [N]umerical Probability." << endl
					 << ">" << flush;
					 
				//Get user input
			    network_option = getcapch();
			    
			    switch(network_option)
			    {
					case 'L': 
					case 'S': 
					case 'X': 
					case 'O': 
					case 'U': // ?
					break;
					
					case 'N'://Choice: Numerical Problem
					{
						cout << "Use validation set to stop learning [Y/N]: " << flush;
						
						usevalid = getcapch();
					}
					
					case 'C'://Choice: Classifier
					{
						cout << "ENTER number of nodes in hidden layer: " << flush;
						cin >> hiddenNeurons;
						cin.ignore(1);
						
						cout << "ENTER max number of epochs for learning: " << flush;
						cin >> max_epoch;		
						cin.ignore(1);	
						
					}break;
					
					default: break;
				}
			}break;
			
			case 'C'://Choice: Set Learning-Constants
			setlparas(learningParameters); break;
			
			case 'I'://Choice: Initialise Random Seed
			{			
				switch(network_option)
				{
					case 'L':
					case 'S':
					case 'X':
					case 'O': 
						cout << "ENTER [0] to use weight in (R.J.M.) Lecture Notes" << endl
							 << "ELSE weights will be randomly set" << endl
							 << ">" << flush;
					break;
					
					default:
					{
						cout << "ENTER seed used for random weights: " << flush;
					} break;
				}
				
				//Get user input
				cin >> weight_option;
				cin.ignore(1);
				
			}break;
			
			default: 
			break;
		}
		
	}
	
	
	return 0;
}

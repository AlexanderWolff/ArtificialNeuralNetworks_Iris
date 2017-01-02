/*
* 	Header-file for Single AND Multi-layered networks
*
* 	Adapted by Alexander Wolff	20/10/14
*/

#ifndef LAYER_H
#define LAYER_H

#include "library.h"

///<summary>
/// This file contains the class definitions for layered networks
///</summary>

///<summary>
/// Simple-layer with linear-activation
///</summary>
class LinearLayerNetwork {					
	
	protected:
		
		//Grants "MultiLayerNetwork" access to the protected functions
		friend class MultiLayerNetwork;
		
		//Stores amount of inputs
		int numInputs;
		
		//Stores amount of neurons
		int numNeurons;
		
		//Stores amount of weights
		int numWeights;	
		
		//Stores neuron-outputs
		double * outputs;
		
		//Stores deltas
		double * deltas;
		
		//Stores weights
		double * weights;
		
		//Stores changes in weights
		double * deltaWeights;
  
		
		///<summary>
		/// Calculates outputs from weights and inputs
		///
		///<argument="const double Inputs[]"> An array containing inputs to the network</argument>
		///</summary>
		virtual void CalcOutputs (const double Inputs[]);
		
		///<summary>
		/// Copies the calculated network outputs into the nth output in data.
		///
		///<argument="int n"> Directs where to store outputs</argument>
		///<argument="dataset &data"> Pointer to the dataset</argument>
		///</summary>
		virtual void StoreOutputs (int n, dataset &data);
		
		///<summary>
		/// Calculates the deltas from the errors
		///
		///<argument="const double Errors[]"> Array containing the errors between targets and outputs</argument>
		///</summary>
		virtual void FindDeltas (const double Errors[]);
		
		///<summary>
		/// Changes all the weights in the layer using inputs, deltas, learning rate and momentum
		///
		///<argument="const doublt Inputs[]"> Array containing the inputs to the layer</argument>
		///<argument="const double learningParameters[]"> Array containing the parameters: {learning-rate, momentum}</argument>
		///</summary>
		virtual void ChangeAllWeights (const double Inputs[], const double learningParameters[]);
		
		///<summary>
		/// Calculates the weighted sum of deltas in this layer, which are the errors in the previous layer
		/// Process known as back-propagation
		///
		///<argument="double previousErrors[]"> Array storing the previous layer's errors</argument>
		///</summary>
		void PrevLayersErrors (double previousErrors[]);
	
	public:
		
		///<summary>
		/// Constructor
		///
		///<argument="int numInputs"> Amount of inputs for the new layer</argument>
		///<argument="int numOutputs"> Amount of outputs for the new layer</argument>
		///</summary>
		LinearLayerNetwork (int numInputs, int numOutputs);
		
		///<summary>
		///Destructor
		///</summary>
		virtual ~LinearLayerNetwork ();
		
		///<summary>
		/// Passes the whole dataset to the network, calculates outputs and stores them in the dataset
		///
		///<argument="dataset &data"> Pointer to the dataset</argument>
		///</summary>
		virtual void ComputeNetwork (dataset &data);
		
		///<summary>
		/// Passes the whole dataset to the network, calculates outputs, stores them in data 
		/// and adjusts the weights using the delta rule
		/// 
		///<argument="dataset &data"> Pointer to the dataset</argument>
		///<argument="const double learningParameters[]"> Array containing the parameters: {learning-rate, momentum}</argument>
		///</summary>
		virtual void AdaptNetwork (dataset &data, const double learningParameters[]);
		
		///<summary>
		/// Initialises the weights in the network using the values in initialWeights[]
		///
		///<argument="const double initialWeights[]"> Array containing initial weights</argument>
		///</summary>
		virtual void SetTheWeights (const double initialWeights[]);
		
		///<summary>
		/// Returns the amount of weights in this layer as an integer
		///</summary>
		virtual int HowManyWeights ();
		
		///<summary>
		/// Copies the weights in this layer into the argument array "theWeights[]"
		///
		///<argument="double theWeights[]"> Array onto the network's weights are copied</argument>
		///</summary>
		virtual void ReturnTheWeights (double theWeights[]);
};

///<summary>
/// Simple-layer with sigmoidal-activation, inherits most of its methods from the "LinearLayerNetwork" object
///</summary>
class SigmoidalLayerNetwork : public LinearLayerNetwork {
	
	protected:	
		
		///<summary>
		/// Calculates the deltas using the formula:
		/// "Errors * Output * (1 - Output)"
		///
		///<argument="const double Errors[]"> Array containing the errors of the network (target - output)</argument>
		///</summary>
		virtual void FindDeltas (const double Errors[]);
		
		///<summary>
		/// Calculates the outputs using the formula:
		/// Temporary_Output = Input * Weight
		/// Actual_Output = 1 / (1 + exp( - Temporary_Output ) )
		///
		///<argument="const double Inputs[]"> Array containing the inputs to the network</argument>
		///</summary>
		virtual void CalcOutputs (const double Inputs[]);	
		
	public:
	
		///<summary>
		/// Constructor
		///
		///<argument="int numInputs"> Amount of inputs to the network</argument>
		///<argument="int numOutputs"> Amount of outputs to the network</argument>
		///</summary>
		SigmoidalLayerNetwork (int numInputs, int numOutputs); 
		
		///<summary>
		/// Destructor
		///</summary>
		virtual ~SigmoidalLayerNetwork ();			
		
};

///<summary>
/// Multi-layered network with sigmoidal-activation, inherits most of its methods from the "SigmoidalLayerNetwork" object
/// Has a hidden layer and an output layer
///</summary>
class MultiLayerNetwork : public SigmoidalLayerNetwork {
				
	protected:
	
		//Pointer to the next layer
		LinearLayerNetwork *nextlayer;
   
		///<summary>
		/// Calculates outputs from weights and inputs
		///
		///<argument="const double Inputs[]"> An array containing inputs to the network</argument>
		///</summary>
		virtual void CalcOutputs (const double Inputs[]);
		
		///<summary>
		/// Copies the calculated network outputs into the nth output in data.
		///
		///<argument="int n"> Directs where to store outputs</argument>
		///<argument="dataset &data"> Pointer to the dataset</argument>
		///</summary>
		virtual void StoreOutputs (int n, dataset &data);
		
		///<summary>
		/// Calculates the deltas in this layer and the next layer 
		/// using the formula:
		/// "Errors * Output * (1 - Output)"
		///
		///<argument="const double Errors[]"> Array containing the errors of the network (target - output)</argument>
		///</summary>
		virtual void FindDeltas (const double Errors[]);
		
		///<summary>
		/// Changes all the weights in the layer using inputs, deltas, learning rate and momentum
		///
		///<argument="const doublt Inputs[]"> Array containing the inputs to the layer</argument>
		///<argument="const double learningParameters[]"> Array containing the parameters: {learning-rate, momentum}</argument>
		///</summary>
		virtual void ChangeAllWeights (const double Inputs[], const double learningParameters[]);

	public:
	
	///<summary>
	/// Constructor
	///
	///<argument="int numInputs"> Amount of inputs to the layer</argument>
	///<argument="int numOutputs"> Amount of outputs to the layer</argument>
	///<argyment="LinearLayerNetwork *to_next_layer"> Pointer to the next layer</argument>
	///</summary>
	MultiLayerNetwork (int numInputs, int numOutputs, LinearLayerNetwork *to_next_layer); 
	
	///<summary>
	/// Destructor
	///</summary>
	virtual ~MultiLayerNetwork ();			

	///<summary>
	/// Initialises the weights in the main layer and next layers in the network using the values in initialWeights[]
	///
	///<argument="const double initialWeights[]"> Array containing initial weights</argument>
	///</summary>
	virtual void SetTheWeights (const double initialWeights[]);
	
	///<summary>
	/// Returns the amount of weights in this layer and the next as an integer
	///</summary>
	virtual int HowManyWeights ();
	
	///<summary>
	/// Copies the weights of the whole network, therefore of all layers, into the argument array "theWeights[]"
	///
	///<argument="double theWeights[]"> Array onto the network's weights are copied</argument>
	///</summary>
	virtual void ReturnTheWeights (double theWeights[]);
	
};

#endif

/*
* 	Header file for mlpdata 
* 
* 	Contains a class for storing data set for multi-layer-perceptron
* 	inputs and targets can be read from: file or array
* 
* 	Contains also variables for storing calculated outputs and sum squares errors
* 	
* 	[2012] now have data fiels for logic; numerical and classification
*
* 	Adapted by Alexander Wolff 	20/10/14
*/

#ifndef DATA_H
#define DATA_H

#include "library.h"

class dataset {
	int numdataset;
	int numinputs;
	int numoutputs;
	int numinrow;		// is numinputs + 2 * numoutputs
	int datatype;		// 0 for logic, 1 for numerical, 2 for classifier
	double *alldata;	// array for inputs, outputs and targets
	double *mindata;	// array for minimum values of each input/target
	double *maxdata;	// array for maxuimum values of each input/target
	double *errors;		// array for errors of each output : used fro errors, SSE, etc
	double *classifications;  // array for % of correct classifications
	double *scaleddata;	// for rescaling outputs at end for display
	char *dataname;		// name of data
	void GetMemory(char *name);
	void ScaleInsTargets(void);
public:
	dataset();
	dataset(char *filename, char *name);
	dataset(int nin, int nout, int nset, double data[], char *name);
	~dataset();
	double * GetNthInputs (int n);
		// return (address of) array of inputs of nth item in data set
	double * GetNthTargets (int n);
		// return (address of) array of targets of nth item in data set
	double * GetNthOutputs (int n);
		// return address of array of output of nth item in data set
	double * GetNthErrors (int n);
		// return address of of array of errors of nth item in data set
	void SetNthOutputs(int n, double outputs[]);
		// copy actual calculated outputs into nth item in data set
	double * CalcSSE (void);
		// calculate SSE across data set, and return address of array with SSEs for each output
	double TotalSSE (void);
		// calc and return sum of all SSEs of data in set
	double * CalcCorrectClassifications(void);
		// calculate and return address of array of % of correct classifications
	double * CalcScaledData (int n, char which);
		// calculate and return address of scaled version for nth output set
	int numIns (void);
		// return number of inputs
	int numOuts (void);
		// return number of outputs
	int numData (void);
		// return number of data sets
	void printarray (char *s, char which, int n, int nl = 0);
		// print s then specifc array and \n if nl
		// if which is 'I' print inputs; if 'O' print outputs; if 'T print targets,
		//           if 'S' print SSEs, if C print % corerect classifications
		// n specifies nth set of inputs,outputs,targets
	void printdata (int showopt);
		// print aspects of data set.
		//   if showopt = 1, print ins, targets, outs, then SSE / % classification )
	    //   if showopt = 0, just print SSE and (if logic/classifier) % classifications
	void savedata (int goplot = 0);
		// save data set (ins, targets and outs) into file
	    // if goplot, then call tadpole program to plot
};

void dcopy (int num, const double fromarray[], double toarray[]); 
	/// copy num doubles from the fromarray to the toarray


#endif

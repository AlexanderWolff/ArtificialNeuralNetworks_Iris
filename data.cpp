/*
*	Library file for handling datasets for Multi-layered Networks
* 
* 	Adapted by Alexander Wolff	20/10/14
*/


#ifndef DATA_CPP
#define DATA_CPP

#include "Header/library.h"


void dcopy (int num, const double fromarray[], double toarray[]) 
{
	// copy num doubles from the fromarray to the toarray
	for (int i=0; i<num; i++) 
	{
		toarray[i] = fromarray[i];
	}
}


void arrout (char *s, int num, double data[], int nl) 
{
	// routine to output the num values in array data
	// s is a string which precedes the array
	// if nl is true, then \n is then output
	cout.precision (3);				// set precision : format of numbers
	cout << s;						// output s
									// next do num doubles in data
	for (int ct = 0; ct < num; ct++) cout << setw(8) << data[ct];
	if (nl) cout << "\n";			// if desired output newline
}

// Implementation of dataset class

dataset::dataset() 
{
	// argument less constructor, just initialises to 0
	GetMemory("");   // initialise all relevant memory to 0
}

dataset::dataset (char *filename, char *name) {
	// constructor where argument is name of file which contains data
	// this opens files, initialises the number of inputs, etc
	// creates space for the data
	// then reads all the data from the file
	ifstream datafile;				// define stream variable for data

	datafile.open(filename);		// open file of given name
	
	if (datafile.is_open()) {
	datafile >> numinputs >> numoutputs >> numdataset >> datatype;
									// read amounts of data from file and type
	GetMemory(name);				// create space for in/outs/errors etc
	int nd, ndi=0, ct;				// counters

	if (datatype > 0)	{		// if not logic, then read min/max for ins/targets
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read min values inputs and targets
			datafile >> mindata[ct];
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read max inputs and targets
			datafile >> maxdata[ct];
		for (ct=0; ct<numoutputs; ct++) {				// so outputs have same min/max as targets
			mindata[numinputs+numoutputs+ct] = mindata[numinputs+ct];
			maxdata[numinputs+numoutputs+ct] = maxdata[numinputs+ct];
		}
	}

	for (nd=0; nd<numdataset; nd++) {	// read each item from set
	
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read n inputs and targets
			datafile >> alldata[ndi++];
	
		ndi += numoutputs;								// skip passed actual outputs
	}
	ScaleInsTargets();
	datafile.close();					// close file
	}
	else GetMemory("");
}

dataset::dataset (int nin, int nout, int nset, double data[], char *name) {
	// constructor to create dataset where raw data in array 
	// arguments passed numbers of inputs, nin, outputs, nout, and in set, nset
	// data is a large enough array
	numinputs = nin;				// first store data sizes
	numoutputs = nout;
	numdataset = nset;									
	datatype = 0;
	GetMemory(name);					// create space for in/outs/errors etc

	int wd = 0, nd, ndi=0, ct;

	for (nd=0; nd<numdataset; nd++) {	// now read each data item from data
		
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read n inputs and targets
			alldata[ndi++] = data[wd++];
	
		ndi += numoutputs;								// skip passed actual outputs
	}
}

dataset::~dataset () {
				// return memory to heap
	 if (alldata != 0) delete [] alldata;
	 if (errors != 0) delete [] errors;
	 if (scaleddata != 0) delete [] scaleddata;
	 if (dataname != 0) delete [] dataname; 
}

void dataset::GetMemory(char *name) {
		// create dynamic arrays for inputs, outputs, targets and SSEs
	if (strlen(name)>0) {    // if valid data name, initialise memory
		numinrow = numinputs + 2 * numoutputs;			// ie inputs, targets, outputs
		alldata = new double [numinrow * numdataset];	// get memory for all data
		errors = new double [numoutputs];					// and for SSEs
		classifications = new double [numoutputs];		// and for % classifications
		scaleddata = new double [numinrow];				// and for re-Scaled data
		mindata = new double [numinrow];	// for min of all ins/targets/outputs
		maxdata = new double [numinrow];	// for max of all ins/targets/outputs
		for (int ct=0; ct<numinputs + numoutputs; ct++)	{	// set min/max to 0 so no scaling
			mindata[ct] = 0;
			maxdata[ct] = 0;
		}
		dataname = new char [strlen(name)+1];
		strcpy(dataname, name);		// give name of data 
    }
	else {     // Just initialise all to null
		numinputs = 0;
		numoutputs = 0;
		numdataset = 0;
		mindata = 0;
		maxdata = 0;
		numinrow = numinputs + 2 * numoutputs;
		alldata = 0;
		errors = 0;
		scaleddata = 0;
		dataname = 0;
	}
}

void dataset::ScaleInsTargets(void) {
	int ndi = 0;
	for (int nd=0; nd<numdataset; nd++) {	// read each item from set
	
		for (int ct=0; ct<numinputs + numoutputs; ct++)	{	// read n inputs and targets
			if (maxdata[ct] > mindata[ct])

				alldata[ndi] = 0.1 + 0.8 * (alldata[ndi] - mindata[ct]) / (maxdata[ct] - mindata[ct]);
			ndi++;
		}
		ndi += numoutputs;								// skip passed actual outputs
	}

}

double * dataset::GetNthInputs (int n) {
		// return address of (first) input of nth item in data set
	return &alldata[n * numinrow];
}

double * dataset::GetNthTargets (int n){
		// return address of (first) target of nth item in data set
	return &alldata[(n * numinrow) + numinputs];
}

double * dataset::GetNthOutputs (int n){
		// return address of (first) output of nth item in data set
	return &alldata[(n * numinrow) + numinputs + numoutputs];
}

void dataset::SetNthOutputs(int n, double outputs[]) {
		// copy actual calculated outputs into nth item in data set
	dcopy (numoutputs, outputs, GetNthOutputs(n) );
}

double * dataset::GetNthErrors (int n){
		// calculate and return sum of square of errors (targets-outouts)
		// for each output
  double *cops;	// pointer to outputs
  double *ctars;	// and to targets
  int ct; 

	cops = GetNthOutputs(n);
	ctars = GetNthTargets(n);
	for (ct=0; ct<numoutputs; ct++) 
			errors[ct] = *ctars++ - *cops++;	// calc next error
	return &errors[0];
}

double sqr (double v) { return v*v; }		// calculate v^2

double * dataset::CalcSSE (void){
		// calculate and return sum of square of errors (targets-outouts)
		// for each output
  double *cops;	// pointer to outputs
  double *ctars;	// and to targets
  int ct, nct; 

    for (ct=0; ct<numoutputs; ct++) 
		errors[ct] = 0;							// errors = 0
	for (nct=0; nct<numdataset; nct++) {
		cops = GetNthOutputs(nct);
		ctars = GetNthTargets(nct);
		for (ct=0; ct<numoutputs; ct++) 
			errors[ct] += sqr(*cops++ - *ctars++);	// add next SE
	}
    for (ct=0; ct<numoutputs; ct++) 
		errors[ct] = errors[ct] / numdataset;			// divide by num in set
	return &errors[0];
}

double * dataset:: CalcCorrectClassifications(void) {
		// calculate and return sum of square of errors (targets-outouts)
		// for each output
  double *cops;	// pointer to outputs
  double *ctars;	// and to targets
  int ct, nct; 

    for (ct=0; ct<numoutputs; ct++) 
		classifications [ct] = 0;							// correct classifications = 0
	for (nct=0; nct<numdataset; nct++) {
		dcopy(numoutputs, CalcScaledData(nct, 'O'), &errors[0]);	// put scaled outputs in errors array
		cops = &errors[0];											// point to it
		ctars = CalcScaledData(nct, 'T');						// point to scaled targets
		for (ct=0; ct<numoutputs; ct++) 
			if (abs (*cops++ - *ctars++) < 0.001) classifications[ct] += 1;	// if scaled Target ~ Scaled Output
	}
    for (ct=0; ct<numoutputs; ct++) 
		classifications[ct] = 100 * classifications[ct] / numdataset;			// divide by num in set
	return &classifications[0];
}

double * dataset::CalcScaledData(int n, char which) {
double *dataline = GetNthInputs(n);
int minnum, maxnum;
	switch (which) {
	   case 'I' :  minnum = 0; maxnum = numinputs; break;
	   case 'T' :  minnum = numinputs; maxnum = minnum + numoutputs; break;
	   case 'O' :  minnum = numinputs+numoutputs; maxnum = minnum + numoutputs; break;
	   case 'A' :  minnum = 0; maxnum = numinrow; break;
	} 
	for (int ct=minnum; ct<maxnum; ct++) {
		scaleddata[ct-minnum] = dataline[ct];
		if (datatype == 0) {
			if (ct>=numinputs) {
			   if (scaleddata[ct-minnum]<= 0.5) scaleddata[ct-minnum] = 0; else scaleddata[ct-minnum] = 1;
			}
		}
		else {
			if (maxdata[ct] > mindata[ct]) 
			   scaleddata[ct-minnum] = mindata[ct] + (scaleddata[ct-minnum]-0.1) * (maxdata[ct]-mindata[ct]) / 0.8;
			if ( (datatype == 2) && (ct>=numinputs) )
				scaleddata[ct-minnum] = floor(0.5+scaleddata[ct-minnum]);
		}
	}
	return &scaleddata[0];
}

double dataset::TotalSSE (void) {
		// calc and return sum of all SSEs of data in set
	double ans = 0;
	CalcSSE();
	for (int ct=0; ct<numoutputs; ct++) ans += errors[ct];
	return ans;
}

int dataset::numIns(void) {
		// return number of data sets
	return numinputs;
}

int dataset::numOuts(void) {
		// return number of data sets
	return numoutputs;
}

int dataset::numData(void) {
		// return number of data sets
	return numdataset;
}

void dataset::printarray (char *s, char which, int n, int nl) {
		// print s then specifc array and \n if nl
		// if which is 'I' print inputs; if 'O' print outputs; 
		//          if 'T print targets, if 'S' print SSEs
		// n specifies nth set of inputs,outputs, targets
	switch (which) {
	case 'i': 
	case 'I' : arrout(s, numinputs, GetNthInputs(n), nl); break;
	case 'o' :
	case 'O' : arrout(s, numoutputs, GetNthOutputs(n), nl); break;
	case 't' :
	case 'T' : arrout(s, numoutputs, GetNthTargets(n), nl); break;
	case 's' :
	case 'S' : arrout(s, numoutputs, CalcSSE(), nl); break;
	case 'c' :
	case 'C' : arrout(s, numoutputs, CalcCorrectClassifications(), nl); break;
	case 'r' :
	case 'R' : arrout(s, numoutputs, CalcScaledData(n, 'O'), nl); break;
	}
}

void dataset::printdata (int showopt) {
	// pass a training set in data to network, show results
	if (showopt > 0) {
		cout << setw(1 + 8*numinputs) << "Inputs" 
			 << setw(3 + 8*numoutputs) << "Targets"
			 << setw(3 + 8*numoutputs) << " Actuals" 
			 << setw(3 + 8*numoutputs) << "Rescaled\n";
	  for (int ct=0; ct<numdataset; ct++) {
		printarray (" ", 'I', ct, 0);
		printarray (" : ", 'T', ct, 0);
		printarray (" : ", 'O', ct, 0);
		printarray (" : ", 'R', ct, 0);
		cout << "\n";
	  }
	}
	else  cout << dataname << " : ";
	if (showopt >= 0 ) {
	  if (datatype < 2) printarray ("Mean Sum Square Errors are ", 'S', 0, 1);
	  if ( (datatype == 2) || ( (datatype == 0) && (showopt > 0) ) ) printarray("% Correct Classifications ", 'C', 0, 1);
	}

}

void dataset::savedata (int goplot) {
		// save data set (ins, targets and outs)
	ofstream datafile;				// define stream variable for data
	int ct2;
	char temp[80];
	strcpy(temp, dataname);
	strcat(temp, "full.txt");
	datafile.open(temp);		// open file of given name
	if (datafile.is_open())   {
      datafile << numinputs << " " << numoutputs << " " << numdataset << "\n";
      for (int ct=0; ct<numdataset; ct++) {
	    CalcScaledData(ct, 'A');
		for (ct2=0; ct2<numinrow; ct2++) datafile << scaleddata[ct2] << "\t";
		datafile << "\n";
	  } 
	  datafile.close();
	  if (goplot) {		// evoke tadpole.exe with name of file
		  cout << "Invoke the tadpole program, select file " << dataname << "full.txt and plot response.\n" ;
	  }
	}
	else cout << "Unable to create " << temp << "\n";
}

#endif

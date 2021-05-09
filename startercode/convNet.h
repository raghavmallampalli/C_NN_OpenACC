#ifndef CONVNET_H
#define CONVNET_H

typedef struct ConvNetInput
{
    int convImageDim;
    int convKernelDim;
    int **convImage;
    int **convKernel;
    int convStride;
    int fullyConnWeightsNum;
    int *fullyConnWeights;
} ConvNetInput;


/*
	Add all the structures required for the program here
*/

struct args1
{
	int startpos[2];
	ConvNetInput C_in;
};

struct args2
{
	int K;
	int **out;
};

struct args3
{
	int posi, posj;
	int **out;
	int *OneDOut;
};

struct argsY
{
	int* final;
	int* wts;
	int n;
};

int convNet(char *fileName);
void getData(FILE *inputFile, ConvNetInput *ConvNetInput);

/* 
	Add declarations of all the required functions here
*/

void *printinput(void* param);
void *CLrunner(void* param);
void *printCLOut(void* param);
void *onedrunner(void* param);
void *findY(void* param);

#endif
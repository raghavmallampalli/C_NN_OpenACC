#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include "convNet.h"

void getData(FILE *inputFile, ConvNetInput *convNetInput)
{
    // Image input.
    fscanf(inputFile, "%d", &convNetInput->convImageDim);
    convNetInput->convImage = malloc(sizeof(int *) * convNetInput->convImageDim);
    for (int i = 0; i < convNetInput->convImageDim; i++)
    {
        (convNetInput->convImage)[i] = (int *) malloc(sizeof(int) * convNetInput->convImageDim);
        for (int j = 0; j < convNetInput->convImageDim; j++)
        {
            fscanf(inputFile, " %d", &convNetInput->convImage[i][j]);
        }
    }

    // Kernel Input.
    fscanf(inputFile, "%d", &convNetInput->convKernelDim);
    convNetInput->convKernel = malloc(sizeof(int *) * convNetInput->convKernelDim);
    for (int i = 0; i < convNetInput->convKernelDim; i++)
    {
        convNetInput->convKernel[i] = (int *) malloc(sizeof(int) * convNetInput->convKernelDim);
        for (int j = 0; j < convNetInput->convKernelDim; j++)
        {
            fscanf(inputFile, " %d", &convNetInput->convKernel[i][j]);
        }
    }

    fscanf(inputFile, "%d", &convNetInput->convStride);

    // Calculate the number of weights required for the fully-connected layer.
    int sqrRootWeightsNum = ((convNetInput->convImageDim) - (convNetInput->convKernelDim)) / (convNetInput->convStride) + 1;
    convNetInput->fullyConnWeightsNum = sqrRootWeightsNum * sqrRootWeightsNum;
    convNetInput->fullyConnWeights = (int *) malloc(sizeof(int) * convNetInput->fullyConnWeightsNum);
    for (int i = 0; i < convNetInput->fullyConnWeightsNum; i++)
    {
        fscanf(inputFile, " %d", &(convNetInput->fullyConnWeights)[i]);
    }

    return;
}

int convNet(char* filename)
{
	FILE *fpi;
	fpi = fopen(filename, "r");
	
	int i,j,Y,K;
	struct ConvNetInput Cin;
	getData(fpi, &Cin);
	K=(Cin.convImageDim-Cin.convKernelDim+Cin.convStride)/Cin.convStride;
	int *res_ptr;
	pthread_attr_t attr1, attr2;	
	pthread_attr_init(&attr1);
	pthread_attr_setdetachstate(&attr1, PTHREAD_CREATE_DETACHED);
	
	pthread_t tid0;
	pthread_t tidCL[K][K], tid00, tid[K], tidfin;
	
	struct args1 x1[K][K];
	struct args2 x2;
	struct argsY y1; 
	pthread_create(&tid0, &attr1, printinput, &Cin);
	pthread_join(tid0,NULL);


	int** CLOut = (int **) malloc(sizeof(int *) * K);
	for(i=0;i<K;i++)
	{
		CLOut[i] = (int *) malloc(sizeof(int) * K);
		for(j=0;j<K;j++)
		{
			x1[i][j].C_in = Cin;
			x1[i][j].startpos[0] = Cin.convStride * i; 
			x1[i][j].startpos[1] = Cin.convStride * j;  
			pthread_create(&tidCL[i][j], NULL, CLrunner, &x1[i][j]);
		}
	}
	for(i=0;i<K;i++)
		for(j=0;j<K;j++)
		{
			pthread_join(tidCL[i][j], (void **) &res_ptr);
			CLOut[i][j] = *res_ptr;
		}
	x2.K = K;
	x2.out = malloc(sizeof(int *) * K);
	for(i=0;i<K;i++)
	{
		x2.out[i] = malloc(sizeof(int) * K);
		for(j=0;j<K;j++)
			x2.out[i][j] = CLOut[i][j];
	}
	pthread_attr_init(&attr2);
	pthread_create(&tid00, &attr2, printCLOut, &x2);
	int detstate;
	if(pthread_attr_getdetachstate(&attr2, &detstate)==PTHREAD_CREATE_JOINABLE) pthread_join(tid00, NULL);
	
	int n = K*K;
	struct args3 x3[n];
	
	int* OneDOut =  (int *) malloc(sizeof(int) * n);
	for(i=0;i<n;i++)
	{	
		x3[i].posi = i/K;
		x3[i].posj = i%K;
		x3[i].out = CLOut; //
		pthread_create(&tid[i], NULL, tid[i], &x3[i]);
		OneDOut[i] = *(x3[i].OneDOut);
	}
	
	for(i=0;i<K;i++)
		pthread_join(tid[i], NULL);
	

	y1.n = n;
	y1.final = OneDOut;//
	for(i=0;i<n;i++)
	{
		y1.wts[i] = Cin.fullyConnWeights[i];
	}
	pthread_create(&tidfin, NULL, findY, &y1);
	pthread_join(tidfin, &res_ptr);
	Y = *res_ptr;
	free(res_ptr);
	free(OneDOut);
	free(CLOut);
	fclose(fpi);
	return Y; 
}

void *printinput(void* param)
{
	struct ConvNetInput *x = (struct ConvNetInput *) param;
	int i,j;
	printf("convImageDim: %d\n",x->convImageDim);
	printf("Image input:\n");
	for(i=0;i<x->convImageDim;i++)
	{	for(j=0;j<x->convImageDim;j++)
			printf("%d ", x->convImage[i][j]);
		printf("\n");
	}
	printf("\n");
	printf("convKernelDim: %d\n",x->convKernelDim);
	printf("Kernel input:\n");
	for(i=0;i<x->convKernelDim;i++)
	{	for(j=0;j<x->convKernelDim;j++)
			printf("%d ", x->convKernel[i][j]);
		printf("\n");
	}
	printf("\n");
	printf("convStride: %d\n\n",x->convStride);
	printf("fullyConnWeightsNum: %d\n\n",x->fullyConnWeightsNum);
	printf("fullyConnWeights:\n\n");
	for(i=0;i<x->fullyConnWeightsNum;i++)
		printf("%d ",x->fullyConnWeights[i]);
	printf("\n\n");
}

void *CLrunner(void *param)
{
	struct args1 *x = (struct args1 *) param;
	int i,j;
	int m = x->C_in.convKernelDim;
	int s0 = x->startpos[0];
	int s1 = x->startpos[1];
	int *result = malloc(sizeof(int));
	*result = 0;
	for(i=s0;i<s0+m;i++)
		for(j=s1;j<s1+m;j++)
			*result += ((x->C_in.convImage[i][j])*(x->C_in.convKernel[i-s0][j-s1]));
	pthread_exit(result);
}

void *printCLOut(void *param)
{
	struct args2 *x = (struct args2 *) param;
	int i,j;

	printf("Thread is currently joinable, should it be converted to detached state? (Y/n): "); 
	char option;
	scanf("%c", &option);
	if(option=='y'||option=='Y')
		pthread_detach(pthread_self());
	else if(option=='n'||option=='N')
		printf("Thread has continued to stay joinable.");

	printf("Convolution Output:\nOutput dim: %dx%d\nOutput Matrix:\n",x->K,x->K);
	for(i=0;i<x->K;i++)
	{	for(j=0;j<x->K;j++)
		{
			printf("%d ",x->out[i][j]);
		}
		printf("\n");
	}
}

void *onedrunner(void *param)
{
	struct args3 *x = (struct args3 *) param;
	x->OneDOut = malloc(sizeof(int));
	*(x->OneDOut) = x->out[x->posi][x->posj];
}

void *findY(void *param)
{
	struct argsY *x = (struct argsY *) param;
	int i;
	int *result = malloc(sizeof(int));
	*result = 0;
	for(i=0;i<x->n;i++)
		*result += ((x->final[i])*(x->wts[i]));
	pthread_exit(result);
}
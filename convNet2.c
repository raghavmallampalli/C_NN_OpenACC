#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

/*
 
Raghav's Copy

Program to pre-process an image using convolution and predict the class of the image using neural net.

Ensure input is a file with tab separated pixel values with equal width and height.
Dimension should also be a power of 2 greater than or equal to 128.

Input image dimension as image_size

Architecture:
Input > Edge detection convolution > Average Pooling > Sharpen convolution > Average Pooling > 
Custom convolution > Average Pooling > Fully connected hidden layer > Output

256 > 256 > 64 > 64 > 16 > 16 > 4 (flattened to 16) >
20 > 7 (output)
*/

// conv macros
#define image_size 256
#define conv_1_size 64
#define conv_2_size 16
#define final_2d_size 4

// NN macros 
#define N 5
#define input_size 57
#define no_epoch 10000
#define lr 0.01
#define hidden_nodes 4
#define output_labels 7
#define nfeatures 16

typedef struct ConvLayer
{
    int InputDim;
    int **Kernel;
} ConvLayer;

int** getImage(FILE *inputFile)
{
	/*
	Get image from file and save to array.
	
	Arguments:
	FILE *inputFile		Tab separated pixel values file (rows as rows)
	
	Returns:
	int **input		 	image pixel values, 2D array [image_size][image_size]
	*/
	int** input;
	input = (int **) malloc(sizeof(int *) * image_size);
    for (int i = 0; i < image_size; i++)
    {
        input[i] = (int *) malloc(sizeof(int) * image_size);
        for (int j = 0; j < image_size; j++)
        {
            fscanf(inputFile, " %d", &input[i][j]);
			//printf("%d,",input[i][j]);
        }
    }

	return input;
}

int** getLabels(FILE *inputFile)
{
	/*
	Get image from file and save to array.
	
	Arguments:
	FILE *inputFile		Tab separated pixel values file (rows as rows)
	
	Returns:
	int **input		 	image pixel values, 2D array [image_size][image_size]
	*/
	int** input;
	input = (int **) malloc(sizeof(int *) * 5766);
    for (int i = 0; i < 5766; i++)
    {
        input[i] = (int *) malloc(sizeof(int) * 7);
        for (int j = 0; j < 7; j++)
        {
            fscanf(inputFile, " %d", &input[i][j]);
        }
    }

	return input;
}

void setEdgeConv(ConvLayer *convInput, int size)
{
	/*
	Generate instance of structure with an edge detection kernel and input size
	
	Arguments:
	ConvLayer *convInput		
	int size					Input size of convolution
	Returns:
	-
	*/

	// Dimension
    convInput->InputDim = size;

    // Kernel Input.
    convInput->Kernel = malloc(sizeof(int *) * 3);
    for (int i = 0; i < 3; i++)
        convInput->Kernel[i] = (int *) malloc(sizeof(int) * 3);
	
	// Kernel set
    (convInput->Kernel)[0][0] = -1;
	(convInput->Kernel)[0][1] = 0;
	(convInput->Kernel)[0][2] = 1;
	(convInput->Kernel)[1][0] = -2;
	(convInput->Kernel)[1][1] = 0;
	(convInput->Kernel)[1][2] = 2;
	(convInput->Kernel)[2][0] = -1;
	(convInput->Kernel)[2][1] = 0;
	(convInput->Kernel)[2][2] = 1;

    return;
}

void setSharpenConv(ConvLayer *convInput, int size)
{
	/*
	Generate instance of structure with a sharpen kernel and input size
	
	Arguments:
	ConvLayer *convInput		
	int size					Input size of convolution
	Returns:
	-
	*/

	// Dimension
    convInput->InputDim = size;

    // Kernel Input.
    convInput->Kernel = malloc(sizeof(int *) * 3);
    for (int i = 0; i < 3; i++)
        convInput->Kernel[i] = (int *) malloc(sizeof(int) * 3);
	
	// Kernel set
    (convInput->Kernel)[0][0] = 0;
	(convInput->Kernel)[0][1] = -1;
	(convInput->Kernel)[0][2] = 0;
	(convInput->Kernel)[1][0] = -1;
	(convInput->Kernel)[1][1] = 5;
	(convInput->Kernel)[1][2] = -1;
	(convInput->Kernel)[2][0] = 0;
	(convInput->Kernel)[2][1] = -1;
	(convInput->Kernel)[2][2] = 0;

    return;
}


void setHyperParamConv(ConvLayer *convInput, int size)
{
	/*
	Generate instance of structure with a custom kernel and input size
	
	Arguments:
	ConvLayer *convInput		
	int size					Input size of convolution
	Returns:
	-
	*/

	// Dimension
    convInput->InputDim = size;

    // Kernel Initialize
    convInput->Kernel = malloc(sizeof(int *) * 3);
    for (int i = 0; i < 3; i++)
        convInput->Kernel[i] = (int *) malloc(sizeof(int) * 3);
	
	// Kernel set
    (convInput->Kernel)[0][0] = -1;
	(convInput->Kernel)[0][1] = -1;
	(convInput->Kernel)[0][2] = -1;
	(convInput->Kernel)[1][0] = -1;
	(convInput->Kernel)[1][1] = 8;
	(convInput->Kernel)[1][2] = -1;
	(convInput->Kernel)[2][0] = -1;
	(convInput->Kernel)[2][1] = -1;
	(convInput->Kernel)[2][2] = -1;

    return;
}

int** averagePooling(int **input, int inputSize)
{
	/*
	Perform average pooling of input, reduce each dimension by 4 times.
	
	Arguments:
	int **input			input data, 2D array [inputSize][inputSize]
	int inputSize		Input size of pooling
	Returns:
	int **output		output data, 2D array [inputSize/4][inputSize/4]
	*/

	int outputSize = inputSize/4;
	int** output = (int **) malloc(sizeof(int *) * outputSize);
	for(int i=0;i<outputSize;i++)
		output[i] = (int *) malloc(sizeof(int) * outputSize);

	int k=0,l;

	for(int i=0;i<inputSize;i+=4)
	{
		l=0;
		for(int j=0;j<inputSize;j+=4)
		{
			output[k][l] = 0.25*(input[i][j]+input[i+1][j]+input[i][j+1]+input[i+1][j+1]);
			l=l+1;
		}
		k=k+1;
	}
	return output;
}

int** convolve(ConvLayer C, int **input)
{
	/*
	Perform convolution of input.
	Input and output are of same dimension: borders are set to 0
	Kernel used is 3x3
	
	Arguments:
	ConvLayer *convInput		

	int **input				input data, 2D array [C.InputDim][C.InputDim]
	Returns:
	int **output			output data, 2D array [C.InputDim][C.InputDim]
	*/

	int** output = (int **) malloc(sizeof(int *) * C.InputDim);
	for(int i=0;i<C.InputDim;i++)
		output[i] = (int *) malloc(sizeof(int) * C.InputDim);

	for(int i=0;i<C.InputDim;i++)
	{
		for(int j=0;j<C.InputDim;j++)
		{	
			if ( i!=0 && j!=0 && i!=(C.InputDim-1) && j!=(C.InputDim-1) )
			{
				output[i][j] = (
					(C.Kernel)[0][0]*input[i-1][j-1] +
					(C.Kernel)[0][1]*input[i-1][j] +
					(C.Kernel)[0][2]*input[i-1][j+1] +
					
					(C.Kernel)[1][0]*input[i][j-1] +
					(C.Kernel)[1][1]*input[i][j] +
					(C.Kernel)[1][2]*input[i][j+1] +

					(C.Kernel)[2][0]*input[i+1][j-1] +
					(C.Kernel)[2][1]*input[i+1][j] +
					(C.Kernel)[2][2]*input[i+1][j+1]
				);

			}
			else
				output[i][j]=0; // FIX: Replace with proper calculations if time is there
			//else
			//{
			//	int l,r,lt,t,rt,lb,b,rb;
			//	if (i==0)
			//		l=0;
			//	if (i==(C.InputDim-1))
			//		r=0;
			//	if (j==0)
			//		t=0;
			//	if (j==(C.InputDim-1))
			//		b=0;
			//	
			//	if (l==0 && t==0)
			//		lt=0;
			//	if (l==0 && b==0)
			//		lb=0;
			//	if (r==0 && t==0)
			//		rt=0;
			//	if (r==0 && b==0)
			//		rb=0;
			//}
		}
	}
	return output;

}
double* softmax(double x[]) { 
	/*
	Softmax activation function for output layer.
	*/
	double sum=0.0, num[input_size];
	static double ratio[input_size];
	for (int i = 0; i < input_size; ++i)
	{
		sum+=expf(x[i]);
		num[i] = expf(x[i]);
	}

	for (int i = 0; i < input_size; ++i)
	{
		ratio[i] = num[i]/sum;
	}
	return ratio; 
}

double sigmoid(double x) {
	/*
	Sigmoid activation function for neural network nodes
	*/
	return 1/(1+expf(-x));
}

double dSigmoid(double x) {
	/*
	Derivative of sigmoid function for back propogation calculations.
	*/
	return sigmoid(x)*sigmoid(1-x);
}

double printarr(double x[], int size) {
	for (int i = 0; i < size; ++i)
	{
		printf("%f\n",x[i]);
	}
	printf("\n");
	return 0;
}

int main(){
	printf("blah\n");
	char image_files[1733*6][30];
    int k=0;
    for(int j=1;j<1733;j++)
        for(int i=0;i<6;i++)
        sprintf(image_files[k++],"%.5d_%d",j,i);

		printf("Defined files.\n");
	double x[input_size][nfeatures];
	int	i=0,j=0,n=0;

	// Setting up convolution layers
	struct ConvLayer edge;
	setEdgeConv(&edge, image_size);
	struct ConvLayer sharp;
	setSharpenConv(&sharp, conv_1_size);
	struct ConvLayer manual;
	setHyperParamConv(&manual, conv_2_size);
	
	FILE *fpi;
	int file_counter=0,tries=0;
	printf("Pre-processing data...\n");
	for(n=0;n<input_size;n++)
	{
		// Get image (input)
		char path[45];
		strcpy(path,"image_csv_files/");
		strcat(path,image_files[file_counter++]);
		strcat(path,".csv");
		while(1) {
			if(access(path,R_OK)==0) {
				//printf("Processing file: %s\n", path);
				fpi = fopen(path,"r");
				break;
			}
			else {
				//printf("Skipping file: %s\n", path);
				strcpy(path,"image_csv_files/");
				strcat(path,image_files[file_counter++]);
				strcat(path,".csv");
			}
		}
		int** image = getImage(fpi);

		// 1st layer output
		int** i1 = convolve(edge, image);
		
		//FILE *fpi1;
		//fpi1 = fopen("edge_out.csv", "w");
		//for(i=0;i<image_size;i++)
		//{
		//	for(j=0;j<image_size;j++)
		//	{
		//		fprintf(fpi1,"%d\t",i1[i][j]);
		//	}
		//	fprintf(fpi1,"\n");
		//}
		//fclose(fpi1);

		// 2nd layer output
		int** i2 = averagePooling(i1, image_size);

		// 3rd layer output
		int** i3 = convolve(sharp, i2);

		// 4th layer output
		int** i4 = averagePooling(i3, conv_1_size);
		
		// 5th layer output
		int** i5 = convolve(manual, i4);
		
		// 6th layer output
		int** i6 = averagePooling(i5, conv_2_size);
	
		//FILE *fpi2;
		//fpi2 = fopen("preproc_out.csv", "w");

		k=0;
		for(i=0;i<final_2d_size;i++)
		{
			for(j=0;j<final_2d_size;j++)
			{
				x[n][k]=i6[i][j];
				//fprintf(fpi2,"%d\t",x0[n][k]);
				k++;
			}
		}

		//fclose(fpi2);
		free(image);	
		free(i1);
		free(i2);
		free(i3);
		free(i4);
		free(i5);
		free(i6);
	}
	fclose(fpi);

	printf("Pre-processing complete. Beginning neural network training.\n\n");

	printf("Processed data:\n");
	for(n=0;n<input_size;n++)
	{
		for(i=0;i<nfeatures;i++)
			printf("%f,",x[n][i]);
		printf("\n");
	}

	FILE* fp_labels = fopen("C_NN_OpenACC/nist_dataset/label_encoding.csv","r");
	int** labels;
	if (!fp_labels)
        printf("Can't open file\n");
    else
    {
		labels = getLabels(fp_labels);	
    }

	fclose(fp_labels);

	// Neural network
	/*
	srand(time(0));


	// Name input (xtrain) x with size input_size*nfeatures and the target labels as label with size input_size*output_labels and
	// use one hot encoding for labels, basically if it belongs to class 1 write it as [1,0,0,0,0,0,0] and so on

	// Feed Forward Neural Network
	double bh[hidden_nodes];
	double bo[output_labels];

	for (int i = 0; i < hidden_nodes; ++i)
	{
		srand(time(0)+rand());
		bh[i] = ((double)rand())/((double)RAND_MAX);
	}

	for (int i = 0; i < output_labels; ++i)
	{
		srand(time(0)+rand());
		bo[i] = ((double)rand())/((double)RAND_MAX);
	}

	printf("Bias1:\n");
	printarr(bh,hidden_nodes);

	printf("Bias2:\n");
	printarr(bo,output_labels); 

	double wh[nfeatures][hidden_nodes];
	double wo[hidden_nodes][output_labels];

	for (int i = 0; i < nfeatures; ++i)
	{
		for (int j = 0; j < hidden_nodes; ++j)
		{
			srand(time(0)+rand());
			wh[i][j] = ((double)rand())/((double)RAND_MAX);
		}
	}

	for (int i = 0; i < hidden_nodes; ++i)
	{
		for (int j = 0; j < output_labels; ++j)
		{
			srand(time(0)+rand());
			wo[i][j] = ((double)rand())/((double)RAND_MAX);
		}
	}

	printf("\n");
	printf("Weights1:\n");
	for (int i = 0; i < nfeatures; ++i)
	{
		for (int j = 0; j < hidden_nodes; ++j)
		{
			printf("%f\t",wh[i][j]);
		}
		printf("\n");
	}

	printf("\n");
	printf("Weights2:\n");
	for (int i = 0; i < hidden_nodes; ++i)
	{
		for (int j = 0; j < output_labels; ++j)
		{
			printf("%f\t",wo[i][j]);
		}
		printf("\n");
	}


	

	//double x[input_size][nfeatures];
	double* soln;

	//for (int i = 0; i < input_size; ++i)
	//{
	//	for (int j = 0; j < nfeatures; ++j)
	//	{
	//		x[i][j] = i+j;
	//	}
	//}

	// soln = softmax(x);
	// printf("Output from Softmax:\n");
	// for (int i = 0; i < input_size; ++i)
	// {
	// 	printf("%f\n",soln[i]);
	// }


	for (size_t epoch = 0; epoch < no_epoch; ++epoch)
	{
		// forward feed
		double zh[input_size][hidden_nodes], ah[input_size][hidden_nodes];
		double zo[input_size][output_labels], ao[input_size][output_labels];

		// computing the hidden layer
		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				zh[i][j] = 0.0;
				for (int k = 0; k < nfeatures; ++k)
				{
					zh[i][j] += (x[i][k]*wh[k][j]) + bh[k];
				}
				ah[i][j] = sigmoid(zh[i][j]);
			}
		}

		double temp[output_labels];
		double* temp2;
		// computing the output layer
		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < output_labels; ++j)
			{
				for (int k = 0; k < hidden_nodes; ++k)
				{
					zo[i][j] += (ah[i][k]*wo[k][j]) + bo[k];
				}
				temp[j] = zo[i][j];
			}
			temp2 = softmax(temp);

			for (int j = 0; j < output_labels; ++j)
			{
				ao[i][j] = temp2[j];
			}
		}


		// Back propogation (cross entropy)
		double dcost_dzo[input_size][output_labels], dzo_dwo[input_size][hidden_nodes], dcost_wo[hidden_nodes][output_labels], dcost_bo[input_size][output_labels];

		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				dzo_dwo[i][j] = ah[i][j];
			}
			for (int j = 0; j < output_labels; ++j)
			{
				dcost_dzo[i][j] = ao[i][j] - labels[i][j];
				dcost_bo[i][j] = ao[i][j] - labels[i][j]; 
			}
		}

		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < output_labels; ++j)
			{
				dcost_wo[i][j] = 0.0;
				for (int k = 0; k < hidden_nodes; ++k)
				{
					dcost_wo[i][k] += dzo_dwo[i][k]*dcost_dzo[k][j];
				}
			}
		}



		double dzo_dah[hidden_nodes][output_labels], dcost_dah[input_size][hidden_nodes];
		double dah_dzh[input_size][hidden_nodes], dzh_dwh[input_size][nfeatures];

		for (int i = 0; i < hidden_nodes; ++i)
		{
			for (int j = 0; j < output_labels; ++j)
			{
				dzo_dah[i][j] = wo[i][j];
			}
		}

		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				dcost_dah[i][j]=0.0;
				for (int k = 0; k < output_labels; ++k)
				{
					dcost_dah[i][j] += dcost_dzo[i][k]*dzo_dah[j][k];
				}
				dah_dzh[i][j] = dSigmoid(zh[i][j]);
			}
		}

		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < nfeatures; ++j)
			{
				dzh_dwh[i][j] = x[i][j];
			}
		}

		double dcost_wh[nfeatures][hidden_nodes];

		for (int i = 0; i < nfeatures; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				dcost_wh[i][j] = 0.0;
				for (int k = 0; k < input_size; ++k)
				{
					dcost_wh[i][j] += dzh_dwh[k][i]*dah_dzh[k][j]*dcost_dah[k][j];
				}
			}
		}

		double dcost_bh[input_size][hidden_nodes];

		for (int i = 0; i < input_size; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				dcost_bh[i][j] = dcost_dah[i][j]*dah_dzh[i][j];
			}
		}

		// Updating Weights
		for (int i = 0; i < nfeatures; ++i)
		{
			for (int j = 0; j < hidden_nodes; ++j)
			{
				wh[i][j] -= lr*dcost_wh[i][j];
			}
		}

		double temp3[hidden_nodes];

		for (int i = 0; i < hidden_nodes; ++i)
		{
			temp3[i] = 0.0;
			for (int j = 0; j < input_size; ++j)
			{
				temp3[i] += dcost_bh[j][i];
			}
			bh[i] -= lr*temp3[i];
		}

		for (int i = 0; i < hidden_nodes; ++i)
		{
			for (int j = 0; j < output_labels; ++j)
			{
				wo[i][j] -= lr*dcost_wo[i][j];
			}
		}

		double temp4[output_labels];

		for (int i = 0; i < output_labels; ++i)
		{
			temp4[i] = 0;
			for (int j = 0; j < input_size; ++j)
			{
				temp4[i] += dcost_bo[j][i];
			}
			bo[i] -= lr*temp4[i];
		}

		if (epoch%100 == 0)
		{
			double loss = 0.0;
			for (int i = 0; i < input_size; ++i)
			{
				for (int j = 0; j < output_labels; ++j)
				{
					loss += labels[i][j]*log(ao[i][j]);
					printf("Loss Function Value at %zu epochs: %f\n", epoch, loss);
				}
			}
		}

	}


	// To predict do the following
	// s1 = sigmoid(dot(input,wh) + bh)
	// s2 = softmax(dot(s,wo) + bo)
	// argmax of s2 is the predicted class

	*/
	return 0;


}
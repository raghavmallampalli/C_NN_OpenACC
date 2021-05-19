#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

/*

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
#define input_size 5766
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
	double sum=0.0, num[output_labels];
	static double ratio[output_labels];
	for (int i = 0; i < output_labels; ++i)
	{
		sum+=exp(x[i]);
		num[i] = exp(x[i]);
	}

	for (int i = 0; i < output_labels; ++i)
	{
		ratio[i] = num[i]/sum;
	}
	return ratio; 
}

double sigmoid(double x) {
	/*
	Sigmoid activation function for neural network nodes
	*/
	return 1/(1+expl(-x));
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
void writeToFile(FILE* fpi, int** layer, int size){
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			fprintf(fpi,"%d\t",layer[i][j]);
		}
		fprintf(fpi,"\n");
	}
}

void feedforward(double x1[][nfeatures],double wh1[][hidden_nodes], double bh1[],
double wo1[][output_labels], double bo1[], double zh1[][hidden_nodes], double ah1[][hidden_nodes],
double zo1[][output_labels], double ao1[][output_labels])
{
    /*
    Performs the forward forward phase of a CNN.

    Input:
    x1[][nfeatures]: Dataset
    wh1[][hidden_nodes]: weights of the hidden layer
    bh1[]: Bias of the hidden layer
    wo1[][hidden_nodes]: weights of the output layer
    bo1[]: Bias of the output layer

    Calculates:
    ah1: Output value for the hidden node
    ao1: Output value for the output node
    */
    for (size_t i = 0; i < input_size; i++)
    {
        for (size_t j = 0; j < hidden_nodes; j++)
        {
            zh1[i][j] = 0.0;
            for (size_t k = 0; k < nfeatures; k++)
            {
                zh1[i][j] += x1[i][k]*wh1[k][j];
            }
            zh1[i][j] += bh1[j];
            ah1[i][j]=sigmoid(zh1[i][j]);
        }
    }

    double temp[output_labels];
    double* temp2;
    for (size_t i = 0; i < input_size; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            zo1[i][j]=0.0;
            for (size_t k = 0; k < hidden_nodes; k++)
            {
                zo1[i][j] += ah1[i][k]*wo1[k][j];
            }
            zo1[i][j] += bo1[j];
            temp[j] = zo1[i][j];
        }
        temp2 = softmax(temp);
        for (size_t j = 0; j < output_labels; j++)
        {
            ao1[i][j] = temp2[j];
        }
        
    }
    // free(temp2);
}

void backprpogation(double x1[][nfeatures], int labels1[][output_labels],
double wh1[][hidden_nodes], double bh1[], double wo1[][output_labels], double bo1[], 
double zh1[][hidden_nodes], double ah1[][hidden_nodes], double zo1[][output_labels], 
double ao1[][output_labels], double dcost_dzo1[][output_labels],
double dzo_dwo1[][hidden_nodes],double dcost_wo1[][output_labels], 
double dcost_bo1[][output_labels], double dzo_dah1[][output_labels], 
double dcost_dah1[][hidden_nodes], double dah_dzh1[][hidden_nodes], 
double dzh_dwh1[][nfeatures], double dcost_wh1[][hidden_nodes],
double dcost_bh1[][hidden_nodes])
{
    // Phase 1
    for (size_t i = 0; i < input_size; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            dcost_dzo1[i][j] = ao1[i][j]-labels1[i][j];
            dcost_bo1[i][j] = dcost_dzo1[i][j];
        }
        for (size_t j = 0; j < hidden_nodes; j++)
        {
            dzo_dwo1[i][j] = ah1[i][j];
        }   
    }

    for (size_t i = 0; i < hidden_nodes; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            dcost_wo1[i][j] = 0.0;
            for (size_t k = 0; k < input_size; k++)
            {
                dcost_wo1[i][j]+=dzo_dwo1[k][i]*dcost_dzo1[k][j];
            }
        }
    }

    // Phase 2
    for (size_t i = 0; i < hidden_nodes; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            dzo_dah1[i][j] = wo1[i][j];
        }
    }

    for (size_t i = 0; i < input_size; i++)
    {
        for (size_t j = 0; j < hidden_nodes; j++)
        {
            dcost_dah1[i][j] = 0.0;
            for (size_t k = 0; k < output_labels; k++)
            {
                dcost_dah1[i][j]+=dcost_dzo1[i][k]*dzo_dah1[j][k];
            }
            dah_dzh1[i][j] = dSigmoid(zh1[i][j]);
            dcost_bh1[i][j] = dcost_dah1[i][j]*dah_dzh1[i][j];
        }
        
        for (size_t j = 0; j < nfeatures; j++)
        {
            dzh_dwh1[i][j] = x1[i][j];
        } 
    }

    for (size_t i = 0; i < nfeatures; i++)
    {
        for (size_t j = 0; j < hidden_nodes; j++)
        {
            dcost_wh1[i][j] = 0.0;
            for (size_t k = 0; k < input_size; k++)
            {
                dcost_wh1[i][j] += dzh_dwh1[k][i]*(dah_dzh1[k][j]*dcost_dah1[k][j]);
            }
        }
    }


    // Updating Weights for each layer
    for (size_t i = 0; i < nfeatures; i++)
    {
        for (size_t j = 0; j < hidden_nodes; j++)
        {
            wh1[i][j] -= lr*dcost_wh1[i][j];
        }
    }

    double temp;
    for (size_t i = 0; i < hidden_nodes; i++)
    {
        temp=0.0;
        for (size_t j = 0; j < input_size; j++)
        {
            temp+=dcost_bh1[j][i];
        }
        bh1[i] -= lr*temp;
    }
    
    for (size_t i = 0; i < hidden_nodes; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            wo1[i][j] -= lr*dcost_wo1[i][j];
        }   
    }

    for (size_t i = 0; i < output_labels; i++)
    {
        temp=0.0;
        for (size_t j = 0; j < input_size; j++)
        {
            temp+=dcost_bo1[j][i];
        }
        bo1[i] -= lr*temp;
    }
}

double loss_function(int labels1[][output_labels], double ao1[][output_labels])
{
    double loss =0.0;
    for (size_t i = 0; i < input_size ; i++)
    {
        for (size_t j = 0; j < output_labels; j++)
        {
            loss+=labels1[i][j]*log(ao1[i][j]);
        }
    }

    return loss;
}

int main(){
	
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
	char image_files[1733*6][30];
    int k=0;
    for(int j=1;j<1733;j++)
        for(int i=0;i<6;i++)
        sprintf(image_files[k++],"%.5d_%d",j,i);

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
				// printf("Processing file: %s\n", path);
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
		//writeToFile(fpi1,i1,image_size);
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

		k=0;
		for(i=0;i<final_2d_size;i++)
		{
			for(j=0;j<final_2d_size;j++)
			{
				x[n][k]=i6[i][j];
				k++;
			}
		}

		free(image);	
		free(i1);
		free(i2);
		free(i3);
		free(i4);
		free(i5);
		free(i6);
	}
	fclose(fpi);

	printf("Feature extracted data:\n");
	for(n=0;n<input_size;n++)
	{
		for(i=0;i<nfeatures;i++)
			printf("%f\t",x[n][i]);
		printf("\n");
	}

	printf("Pre-processing complete. Beginning neural network training.\n\n");
	

	FILE* fp_labels = fopen("nist_dataset/label_encoding.csv","r");
	int** labels;
	if (!fp_labels)
        printf("Can't open file\n");
    else
    {
		labels = getLabels(fp_labels);	
    }

	fclose(fp_labels);

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

	// Uncomment this if you want to use fixed weights and not probablistic weights
	
	// for (size_t i = 0; i < input_size; i++)
	// {
	// 	for (size_t j = 0; j < hidden_nodes; j++)
	// 	{
	// 		wh[i][j] = (double)(i+j)/10;
	// 		bh[j] = (double)j/10;
	// 	}
	// }

	// for (size_t i = 0; i < hidden_nodes; i++)
	// {
	// 	for (size_t j = 0; j < output_labels; j++)
	// 	{
	// 		wo[i][j] = (double)(i+j)/10;
	// 		bo[j] = (double)j/10;
	// 	}
		
	// }
	

	printf("Bias1:\n");
	printarr(bh,hidden_nodes);

	printf("Bias2:\n");
	printarr(bo,output_labels); 

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

    double zh[input_size][hidden_nodes], ah[input_size][hidden_nodes];
	double zo[input_size][output_labels], ao[input_size][output_labels];
	double dcost_dzo[input_size][output_labels], dzo_dwo[input_size][hidden_nodes];
	double dcost_wo[hidden_nodes][output_labels], dcost_bo[input_size][output_labels];
	double dzo_dah[hidden_nodes][output_labels], dcost_dah[input_size][hidden_nodes];
	double dah_dzh[input_size][hidden_nodes], dzh_dwh[input_size][nfeatures];
	double dcost_wh[nfeatures][hidden_nodes];
	double dcost_bh[input_size][hidden_nodes];
	double temp_acc = 0, max_acc = 0;
	int epoch_opt = 0;

    for (int epoch = 0; epoch < no_epoch; epoch++)
    {

		for (size_t i = 0; i < input_size; i++)
    	{
        	for (size_t j = 0; j < hidden_nodes; j++)
        	{
            	zh[i][j] = 0.0;
            	for (size_t k = 0; k < nfeatures; k++)
            	{
                	zh[i][j] += x[i][k]*wh[k][j];
            	}
            	zh[i][j] += bh[j];
            	ah[i][j]=sigmoid(zh[i][j]);
        	}
    	}
		// Correct till here


    	double temp[output_labels];
    	double* temp2;
    	for (size_t i = 0; i < input_size; i++)
    	{
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	zo[i][j]=0.0;
            	for (size_t k = 0; k < hidden_nodes; k++)
            	{
                	zo[i][j] += ah[i][k]*wo[k][j];
            	}
            	zo[i][j] += bo[j];
            	temp[j] = zo[i][j];
        	}
        	temp2 = softmax(temp);
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	ao[i][j] = temp2[j];
				// ao[i][j] = (ao[i][j]<0.000001 && ao[i][j]>0) ? 0.000001:ao[i][j];
        	}
        
    	}
    	// free(temp2);

		// Correct upto here

		// Phase 1
    	for (size_t i = 0; i < input_size; i++)
    	{
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	dcost_dzo[i][j] = ao[i][j]-labels[i][j];
            	// dcost_dzo[i][j] = (dcost_dzo[i][j]<0.000001 && dcost_dzo[i][j]>0) ? 0.000001:dcost_dzo[i][j];
				dcost_bo[i][j] = dcost_dzo[i][j];
        	}
        	for (size_t j = 0; j < hidden_nodes; j++)
        	{
            	dzo_dwo[i][j] = ah[i][j];
				// dzo_dwo[i][j] = (dzo_dwo[i][j]<0.000001 && dzo_dwo[i][j]>0) ? 0.000001:dzo_dwo[i][j];
        	}   
    	}

		// correct till here


    	for (size_t i = 0; i < hidden_nodes; i++)
    	{
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	dcost_wo[i][j] = 0.0;
            	for (size_t k = 0; k < input_size; k++)
            	{
                	dcost_wo[i][j]+=dzo_dwo[k][i]*dcost_dzo[k][j];
            	}
				// dcost_wo[i][j] = (dcost_wo[i][j]<0.000001 && dcost_wo[i][j]>0) ? 0.000001:dcost_wo[i][j];
        	}
    	}

		// correct till here

    	// Phase 2
    	for (size_t i = 0; i < hidden_nodes; i++)
    	{
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	dzo_dah[i][j] = wo[i][j];
				// dzo_dah[i][j] = (dzo_dah[i][j]<0.000001 && dzo_dah[i][j]>0) ? 0.000001:dzo_dah[i][j];
        	}
    	}

		// correct till here

    	for (size_t i = 0; i < input_size; i++)
    	{
        	for (size_t j = 0; j < hidden_nodes; j++)
        	{
            	dcost_dah[i][j] = 0.0;
            	for (size_t k = 0; k < output_labels; k++)
            	{
                	dcost_dah[i][j]+=dcost_dzo[i][k]*dzo_dah[j][k];
            	}
            	dah_dzh[i][j] = dSigmoid(zh[i][j]);
            	dcost_bh[i][j] = dcost_dah[i][j]*dah_dzh[i][j];
				// dah_dzh[i][j] = (dah_dzh[i][j]<0.000001 && dah_dzh[i][j]>0) ? 0.000001:dah_dzh[i][j];
				// dcost_bh[i][j] = (dcost_bh[i][j]<0.000001 && dcost_bh[i][j]>0) ? 0.000001:dcost_bh[i][j];
        	}
        
        	for (size_t j = 0; j < nfeatures; j++)
        	{
            	dzh_dwh[i][j] = x[i][j];
        	} 
    	}

		// correct till here (check dah_dzh) Round off errors?
		double temp_mat[input_size][hidden_nodes];
		for (size_t i = 0; i < input_size; i++)
		{
			for (size_t j = 0; j < hidden_nodes; j++)
			{
				temp_mat[i][j] = dah_dzh[i][j]*dcost_dah[i][j];
				// temp_mat[i][j] = (temp_mat[i][j]<0.000001 && temp_mat[i][j]>0) ? 0.000001:temp_mat[i][j];
			}
		}
		
    	// Issue in this loop
		for (size_t i = 0; i < nfeatures; i++)
    	{
        	for (size_t j = 0; j < hidden_nodes; j++)
        	{
            	dcost_wh[i][j] = 0.0;
            	for (size_t k = 0; k < input_size; k++)
            	{
                	dcost_wh[i][j] += dzh_dwh[k][i]*temp_mat[k][j];// (dah_dzh[k][j]*dcost_dah[k][j]);//dzh_dwh[k][i]*
					// dcost_wh[i][j] = (dcost_wh[i][j]<0.000001 && dcost_wh[i][j]>0) ? 0.000001:dcost_wh[i][j];
            	}
        	}
    	}

		// for (size_t i = 0; i < nfeatures; i++)
		// {
		// 	for (size_t j = 0; j < hidden_nodes; j++)
		// 	{
		// 		printf("%f\t",dcost_wh[i][j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("%f\n",temp_mat[2829][0]);
		// printf("%f\n",dah_dzh[282][0]);
		// printf("%f\n",dcost_dah[2829][0]);


    	// Updating Weights for each layer
    	for (size_t i = 0; i < nfeatures; i++)
    	{
        	for (size_t j = 0; j < hidden_nodes; j++)
        	{
            	wh[i][j] -= lr*dcost_wh[i][j];
        	}
    	}

    	double temp3;
    	for (size_t i = 0; i < hidden_nodes; i++)
    	{
        	temp3=0.0;
        	for (size_t j = 0; j < input_size; j++)
        	{
            	temp3+=dcost_bh[j][i];
        	}
        	bh[i] -= lr*temp3;
    	}
    
    	for (size_t i = 0; i < hidden_nodes; i++)
    	{
        	for (size_t j = 0; j < output_labels; j++)
        	{
            	wo[i][j] -= lr*dcost_wo[i][j];
        	}   
    	}

    	for (size_t i = 0; i < output_labels; i++)
    	{
        	temp3=0.0;
        	for (size_t j = 0; j < input_size; j++)
        	{
            	temp3+=dcost_bo[j][i];
        	}
        	bo[i] -= lr*temp3;
    	}

		// for (size_t i = 0; i < input_size; i++)
		// {
		// 	for (size_t j = 0; j < output_labels; j++)
		// 	{
		// 		printf("%f\t",ao[i][j]);
		// 	}
		// 	printf("\n");
		// }
		


        double loss_val= 0.0;
        // feedforward(x,wh,bh,wo,bo,zh,ah,zo,ao);
        // backprpogation(x, labels, wh, bh, wo, bo, zh, ah, zo, ao, dcost_dzo, dzo_dwo,
        // dcost_wo, dcost_bo, dzo_dah, dcost_dah, dah_dzh, dzh_dwh, dcost_wh, dcost_bh);
        
		double max_val=0.0;
		double count=0.0, acc=0.0;;
		for (size_t i = 0; i < input_size; i++)
		{
			for (size_t j = 0; j < output_labels; j++)
			{
				max_val = (ao[i][j]>max_val) ? ao[i][j]:max_val;
			}
			for (size_t j = 0; j < output_labels; j++)
			{
				if (max_val==ao[i][j] && labels[i][j]==1)
				{
					count+=1;
				}	
			}
		}
		acc = (double)count/input_size;
		max_acc = (acc>temp_acc) ? acc:max_acc;
		epoch_opt = (acc>temp_acc) ? epoch:epoch_opt;
		temp_acc=max_acc;
		
		if (epoch%100==0)
        {
            loss_val = loss_function(labels, ao);
            printf("Loss Function Value at %d epochs: %f\n", epoch, loss_val);
			printf("Accuracy: %f\n",acc);
		}
		if (epoch==no_epoch-1)
		{
			printf("Max Acc = %f @ %d epcohs\n",max_acc,epoch_opt);
		}
		
    }
    

    return 0;
}
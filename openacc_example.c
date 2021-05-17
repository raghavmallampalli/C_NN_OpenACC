#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define n 1000
#define n_gangs 100

double f(double x){
    return sin(5*x);
}

int main(){
    // Declare variables and set up system
    double h;
    h=3.0/n;
    double x[n+1],dfdx[n+1],rhs[n+1],y[n+1],A[n+1][n+1];
    int i,j,k;

    // Initialise system
#pragma acc data create(x[n+1],rhs[n+1],A[:n+1][:n+1]) copyout(rhs[n+1],x[n+1],A)
{
#pragma acc parallel loop num_gangs(n_gangs) present(x[:]) 
    for(i=0; i<=n; i++)
        x[i] = h*(i);
#pragma acc parallel loop num_gangs(n_gangs) present(rhs[:])
    for(int i=1; i<n;i++)
        rhs[i] = 3.0*(f(x[i+1])-f(x[i-1]))/h;
    rhs[0] = ( -5.0*f(x[0])/2.0 + 2.0*f(x[1])   + f(x[2])/2.0 )/h;
    rhs[n] = (  5.0*f(x[n])/2.0 - 2.0*f(x[n-1]) - f(x[n-2])/2.0)/h;
#pragma acc parallel loop collapse(2) num_gangs(n_gangs) present(A[:][:])
    for(i=0; i<=n; i++)
    {
        for(j=0; j<=n; j++)
        {
            if(i==j+1 || j==i+1)
                A[i][j]=1;
            else if(i==j)
                A[i][j]=4;
            else
                A[i][j]=0;
        }
    }
    A[0][0]=1;
    A[0][1]=2;
    A[n][n-1]=2;
    A[n][n]=1;

    for(k=0;k<n;k++)
    {
#pragma acc parallel loop num_gangs(n_gangs) present(A[:][:])

        for(i=k+1;i<=n;i++)
        {
            A[i][k]=A[i][k]/A[k][k];
        }
#pragma acc parallel loop num_gangs(n_gangs) collapse(2) present(A[:][:])

        for(j=k+1;j<=n;j++)
        {
            for(i=k+1;i<=n;i++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
        }
    }
}

    // Inverting L matrix
    y[0]=rhs[0];
    for(int i=1;i<=n;i++)
        y[i]=rhs[i]-A[i][i-1]*y[i-1];

    dfdx[n] = y[n]/A[n][n];
    // Inverting U matrix
    for(int i=n-1;i>=0;i--)
        dfdx[i]=(y[i]-A[i][i+1]*dfdx[i+1])/A[i][i];


    printf("Calculation complete.\n");
    // Displaying and saving data for plotting
    //FILE *data;
    //data=fopen("lud.csv","w");
    //fprintf(data,"x,f,dfdx\n");
    //printf("Writing values to lud.csv. Plot with other utility.\n");
    //for(int i=0;i<=n;i++)
    //    fprintf(data,"%lf,%lf,%lf\n",x[i],f(x[i]),dfdx[i]);
    //fclose(data);
    printf("\n");
    return 0;
}

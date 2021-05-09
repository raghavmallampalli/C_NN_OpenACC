#include <stdio.h>
#include <stdlib.h>

#include "convNet.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Please enter the right number of command-line arguments.\n");
        exit(0);
    }

    int finalValue = convNet(argv[1]);
    printf("The value retured by the convNet function is = %d\n", finalValue);

    return 0;
}
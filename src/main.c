#include "device_query.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) 
{
    if(argc < 2) {
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    if(strcmp(argv[1], "query") == 0) {
        device_query();
    }
    else {
        fprintf(stderr, "%s is not a valid option\n", argv[1]);
    }

    return 0;
}

// #include <stdio.h>

#define THREADS_COUNT 512

__global__ void add(int *numberOne, int *numberTwo, int *addition)
{
    addition[threadIdx.x] = numberOne[threadIdx.x] + numberTwo[threadIdx.x];
}

void random_ints(int* dest, int count)
{
    int counter = 0;
    for (counter = 0; counter < count; ++counter)
    {
        dest[counter] = rand();
    }
}

int main(void)
{
    // The host variables.
    int *host_numberOne, *host_numberTwo, *host_addition;

    // The device variables.
    int *device_numberOne, *device_numberTwo, *device_addition;

    // Size of variable per block
    int size = THREADS_COUNT * sizeof(int);

    // int counter = 0;

    // Allocate memory on device for device variables.
    cudaMalloc((void **)&device_numberOne, size);
    cudaMalloc((void **)&device_numberTwo, size);
    cudaMalloc((void **)&device_addition, size);

    // Initialize host variables;
    host_numberOne = (int *)malloc(size);
    random_ints(host_numberOne, THREADS_COUNT);

    host_numberTwo = (int *)malloc(size);
    random_ints(host_numberTwo, THREADS_COUNT);

    host_addition = (int *)malloc(size);

    // Copy host variables to device memory.
    cudaMemcpy(device_numberOne, host_numberOne, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_numberTwo, host_numberTwo, size, cudaMemcpyHostToDevice);

    // Invoke add kernel.
    add<<<1, THREADS_COUNT>>>(device_numberOne, device_numberTwo, device_addition);

    // Copy device variable to host memory.
    cudaMemcpy(host_addition, device_addition, size, cudaMemcpyDeviceToHost);

    // for(counter = 0; counter < THREADS_COUNT; ++counter)
    // {
    //     printf("%d\t+ %d\t= %d\n", host_numberOne[counter], host_numberTwo[counter], host_addition[counter]);
    // }

    // Clean up, free all device allocated memory.
    free(host_numberOne);
    free(host_numberTwo);
    free(host_addition);
    cudaFree(device_numberOne);
    cudaFree(device_numberTwo);
    cudaFree(device_addition);

    return 0;
}

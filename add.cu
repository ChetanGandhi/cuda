#include<stdlib.h>
#include<stdio.h>

__global__ void add(int *numberOne, int *numberTwo, int *addition)
{
    *addition = *numberOne + *numberTwo;
}

int main(void)
{
    // The host variables.
    int host_numberOne, host_numberTwo, host_addition;

    // The device variables.
    int *device_numberOne, *device_numberTwo, *device_addition;

    // Size of variable
    int size = sizeof(int);

    // Allocate memory on device for device variables.
    cudaMalloc((void **)&device_numberOne, size);
    cudaMalloc((void **)&device_numberTwo, size);
    cudaMalloc((void **)&device_addition, size);

    // Initialize host variables;
    host_numberOne = 2;
    host_numberTwo = 7;

    // Copy host variables to device memory.
    cudaMemcpy(device_numberOne, &host_numberOne, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_numberTwo, &host_numberTwo, size, cudaMemcpyHostToDevice);

    // Invoke add kernel.
    add<<<1, 1>>>(device_numberOne, device_numberTwo, device_addition);

    // Copy device variable to host memory.
    cudaMemcpy(&host_addition, device_addition, size, cudaMemcpyDeviceToHost);

    // Clean up, free all device allocated memory.
    cudaFree(device_numberOne);
    cudaFree(device_numberTwo);
    cudaFree(device_addition);

    return 0;
}

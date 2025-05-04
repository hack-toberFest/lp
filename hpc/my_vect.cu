#include<iostream>
#include<cstdlib>
#include<cuda_runtime.h>
using namespace std;

__global__ void add(int* A, int*B, int*C, int N){
    int cur = blockIdx.x * blockDim.x + threadIdx.x;

    if(cur<N){
        C[cur]=A[cur]+ B[cur];
    }
}

int main(){
    int N= 10;
    int size = N*sizeof(int);

    int *A, *B, *C;
    int *devA, *devB, *devC;

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    cudaMalloc(&devA, size);
    cudaMalloc(&devB, size);
    cudaMalloc(&devC, size);

    srand(time(0));
    for(int i=0; i<N; i++){
        A[i]=(rand()%N);
        B[i]= (rand()%N);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice);

    int blocksize= 256;
    int numblock = (N+blocksize-1)/blocksize;

    add<<<numblock, blocksize>>>(devA, devB, devC, N);

    cudaMemcpy(C, devC, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float duration=0;
    cudaEventElapsedTime(&duration, start, end);

    for(int i=0; i<N;i++)        cout<<A[i]<<" ";
    cout<<endl;
    for(int i=0; i<N;i++)        cout<<B[i]<<" ";
    cout<<endl;
    for(int i=0; i<N;i++)        cout<<C[i]<<" ";
    cout<<endl;

    cout<<"time : "<<duration<<" ms";

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);


}
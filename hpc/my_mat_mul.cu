// #include<bits/stdc++.h>
#include <iostream>
#include <cstdlib> 
#include<cuda_runtime.h>
using namespace std;

__global__ void mul(int* A, int* B, int* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<N && col<N){
        int ans=0;
        for(int i=0; i<N; i++){
            ans+= A[row*N+i] * B[N*i + col];
        }
        C[row*N+col]=ans;
    }
}

int main(){
    int N=16;
    int size = N * N * sizeof(int);

    int* A;
    int* B;
    int* C;
    int* devA;
    int* devB;
    int* devC;

    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);
    
    cudaMalloc(&devA, size);
    cudaMalloc(&devB, size);
    cudaMalloc(&devC, size);

    for(int i=0; i<N; i++){
        for(int j=0; j<N;j++){
            A[i*N+j]=(rand()%N);
            B[i*N+j]=(rand()%N);
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16,16);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);

    mul<<<dimGrid, dimBlock>>>(devA, devB, devC, N);

    cudaMemcpy(C, devC, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float duration=0;
    cudaEventElapsedTime(&duration, start, stop);

    for(int i=0; i<N; i++){
        for(int j=0; j<N;j++){
            cout<<C[i*N+j]<<" ";
        }
        cout<<endl;
    }
    cout<<"time : "<<duration<<" ms"<<endl;

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}
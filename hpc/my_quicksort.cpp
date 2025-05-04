#include<bits/stdc++.h>
#include<mpi.h>
using namespace std;

void quicksort(vector<int>& arr, int left, int right){
    if(left>=right) return;
    int pivot = arr[right];
    int curr=left;
    for(int i=left; i<right; i++){
        if(arr[i]<pivot)    swap(arr[curr++], arr[i]);
    }
    swap(arr[curr], arr[right]);

    quicksort(arr,left,curr-1);
    quicksort(arr,curr+1,right);
}

vector<int> merge(vector<int>& left, vector<int>& right){
    int n1=left.size(), n2=right.size();
    vector<int> ans;
    int i=0,j=0;
    while(i<n1 && j<n2){
        if(left[i]<right[j])    ans.push_back(left[i++]);
        else    ans.push_back(right[j++]);
    }
    while(i<n1) ans.push_back(left[i++]);
    while(j<n2) ans.push_back(right[j++]);
    return ans;
}

int main(int argc, char**argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    int N=10;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size= N/size;
    N = chunk_size * size;

    vector<int>data(N), gathered_data(N);
    vector<int> local_data(chunk_size);
    double start, end=0;

    if(rank==0){
        srand(time(0));
        for(int i=0; i<N; i++){
            data[i]= (rand()%N);
        }
        cout<<" data: ";
        for(auto it: data)    cout<<it<<" ";
        cout<<endl;
        start = MPI_Wtime();
    }

    MPI_Scatter(data.data(), chunk_size, MPI_INT,
                local_data.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);
    
    quicksort(local_data,0,chunk_size-1);

    MPI_Gather(local_data.data(), chunk_size, MPI_INT,
                gathered_data.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    if(rank==0){
        vector<int>result(gathered_data.begin(), gathered_data.begin()+chunk_size);
        for(int i=1; i<size; i++){
            vector<int> next(gathered_data.begin()+i*chunk_size, gathered_data.begin()+ (i+1)*chunk_size);
            result = merge(result, next);
        }    

    end=MPI_Wtime();

    for(auto it: result)    cout<<it<<" ";
    cout<<"\nquicksort time took : "<<(end-start)*1000<<" ms"<<endl;
    }
    MPI_Finalize();
}
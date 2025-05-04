#include<bits/stdc++.h>
#include <omp.h>
using namespace std;
const int n = 20;
vector<int> arr(n,0);

void bubble_sort(vector<int> arr){
    int n = arr.size();
    for(int i=0; i<n; i++){
        int index = i%2;
        #pragma omp parallel for shared(arr,index)
        for(int j=index; j<n-1; j+=2)
        {
            if(arr[j]>arr[j+1]) swap(arr[j],arr[j+1]);
        }
    }

    cout<<" bubble sort: ";
    for(auto it:arr)    cout<<it<<"  ";
}
void merge(vector<int>& arr, int low, int mid, int high){
    int n1 = mid-low+1;
    int n2 = high-mid;

    vector<int> left(n1,0), right(n2,0);

    for(int i=0; i<n1; i++) left[i]= arr[low+i];
    for(int i=0; i<n2; i++) right[i]=arr[mid+1+i];

    int i=0, j=0, k=low;
    while(i<n1 && j<n2){
        if(left[i]<right[j]){
            arr[k]=left[i];
            i++;
            k++;
        }
        else{
            arr[k]=right[j];
            j++;
            k++;
        }
    }

    while(i<n1){
        arr[k++]=left[i++];
    }
    while(j<n2){
        arr[k++]=right[j++];
    }
    
}

void merge_sort(vector<int>& arr, int low, int high)
{
    if(low<high){
        int mid = (low+high)/2;
        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort(arr,low,mid);

            #pragma omp section
            merge_sort(arr,mid+1,high);
        }

        merge(arr,low,mid,high);
    }
}

int main(){
    for(int i=0; i<n; i++){
        arr[i]=(rand()%n);
    }

    double start, end=0;
    start = omp_get_wtime();
    bubble_sort(arr);
    end = omp_get_wtime();
    cout<<"\nbubble sort time : "<<(end-start)*1000<<" ms"<<endl;

    start = omp_get_wtime();
    merge_sort(arr,0,n-1);
    end = omp_get_wtime();
    for(auto it:arr)    cout<<it<<" ";
    cout<<"\n merge sort time : "<<(end-start)*1000<<" ms"<<endl;
}
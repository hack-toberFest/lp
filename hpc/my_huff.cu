#include<iostream>
#include<queue>
#include<map>
#include<cuda_runtime.h>
using namespace std;

struct node{
    char data;
    int freq;
    node* left;
    node* right;
    node(char d, int count){
        data=d;
        freq=count;
        left=right=nullptr;
    }
};

struct compare{
    bool operator()(node* left, node* right){
        return left->freq > right->freq;
    }
};

__global__ void count(int* freq, char* input, int N){
    int cur = blockIdx.x * blockDim.x + threadIdx.x;

    if(cur<N)   atomicAdd(&freq[(unsigned char)input[cur]],1);
}

void get_codes(map<char,string>& mp, int* freq){
    //sort counts in descending order
    priority_queue<node*, vector<node*>, compare> pq;
    for(int i=0; i<256; i++){
        if(freq[i]) pq.push(new node((char)i, freq[i]));
    }
    //build tree
    while(pq.size()>1){
        node* lt = pq.top();
        pq.pop();
        node* rt = pq.top();
        pq.pop();
        node* cur = new node('\0', lt->freq+ rt->freq);
        cur->left = lt;
        cur->right=rt;
        pq.push(cur);
    }
    //get codes
    if(!pq.empty()){
        queue<pair<string, node*>> q;
        q.push({"", pq.top()});
        while(!q.empty()){
            auto it = q.front();
            q.pop();
            string code = it.first;
            node* cur = it.second;

            if(!cur->left && !cur->right)   mp[cur->data]= code;
            if(cur->left)  q.push({code+"0", cur->left});
            if(cur->right)  q.push({code+"1", cur->right});
        }
    }
}

int main(){
    string input = "example text";
    int n = input.size();

    char* dev_input;
    int* dev_freq, freq[256]={0};

    cudaMalloc(&dev_input, n);
    cudaMalloc(&dev_freq, 256*sizeof(int));

    cudaMemcpy(dev_input, input.c_str(), n, cudaMemcpyHostToDevice);
    cudaMemset(dev_freq, 0, 256*sizeof(int));

    count<<<(n+255)/256, 256>>>(dev_freq, dev_input, n);
    cudaMemcpy(freq, dev_freq, 256*sizeof(int), cudaMemcpyDeviceToHost);

    map<char,string> codes;

    get_codes(codes, freq);

    string encoded = "";
    for(auto it:input)  encoded+=codes[it];
    cout<<encoded<<endl;

    cudaFree(dev_input);
    cudaFree(dev_freq);
}

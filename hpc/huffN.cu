#include <iostream>
#include <queue>
#include <unordered_map>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
using namespace std;

struct Node {
    char data;
    int freq;
    Node *left, *right;
    Node(char d, int f) : data(d), freq(f), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(Node* l, Node* r) { return l->freq > r->freq; }
};

__global__ void gpu_count(char* input, int* freq, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) atomicAdd(&freq[(unsigned char)input[i]], 1);
}

void build_tree(int* freq, unordered_map<char, string>& codes) {
    priority_queue<Node*, vector<Node*>, Compare> pq;
    for (int i = 0; i < 256; i++)
        if (freq[i]) pq.push(new Node((char)i, freq[i]));

    while (pq.size() > 1) {
        auto left = pq.top(); pq.pop();
        auto right = pq.top(); pq.pop();
        auto parent = new Node('\0', left->freq + right->freq);
        parent->left = left; parent->right = right;
        pq.push(parent);
    }

    if (!pq.empty()) {
        queue<pair<Node*, string>> q;
        q.push({pq.top(), ""});
        while (!q.empty()) {
            auto item = q.front(); q.pop();  // Get the pair from the front
            Node* node = item.first;  // The Node
            string code = item.second;  // The code string
            if (!node->left && !node->right) codes[node->data] = code;
            if (node->left) q.push({node->left, code + "0"});
            if (node->right) q.push({node->right, code + "1"});
        }
    }
}

int main() {
    string input = "example text";
    int size = input.size();

    // GPU frequency count
    char* d_input; int freq[256] = {0}, *d_freq;
    cudaMalloc(&d_input, size); 
    cudaMalloc(&d_freq, 256*sizeof(int));
    cudaMemcpy(d_input, input.c_str(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, 256*sizeof(int));
    gpu_count<<<(size+255)/256, 256>>>(d_input, d_freq, size);
    cudaMemcpy(freq, d_freq, 256*sizeof(int), cudaMemcpyDeviceToHost);

    // CPU build tree and encode
    unordered_map<char, string> codes;
    build_tree(freq, codes);
    
    string encoded;
    for (char c : input) encoded += codes[c];

    // Output result
    cout << "Encoded: " << (encoded.size() > 100 ? encoded.substr(0, 100) + "..." : encoded) 
         << "\nOriginal: " << size*8 << " bits\nEncoded: " << encoded.size() << " bits\n";

    // Cleanup
    cudaFree(d_input); 
    cudaFree(d_freq);
}
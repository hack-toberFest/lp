#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
const int n = 10000;
const int edges=150000;
vector<int> graph[n];
vector<int> visited(n,0);
vector<int> bfs_traversal;
vector<int> dfs_traversal;

void bfs(int curr){
    queue<int> q;
    q.push(curr);
    visited[curr]=true;

    #pragma omp parallel
    {
        #pragma omp single
        {
            while(!q.empty()){            
                int node = -1;
                #pragma omp critical
                {
                    if(q.size()){
                        node = q.front();
                        q.pop();
                    }
                }

                if(node==-1)    continue;
                
                // visited[node]=true;
                #pragma omp critical
                bfs_traversal.push_back(node);
        
                #pragma omp task firstprivate(node)
                {
                    for(auto it :graph[node]){
                        #pragma omp critical
                        {
                            if (!visited[it]) {
                                visited[it] = true;
                                q.push(it);
                            }
                        }
                    }
                }
                #pragma omp taskwait
            
            }
        }
    }
    
}

void dfs(int curr){
    stack<int> st;
    st.push(curr);
    visited[curr]=true;

    #pragma omp parallel
    {
        #pragma omp single 
        {
            while(!st.empty()){
            
                int node = -1;
                #pragma omp critical
                {
                    if(st.size()){
                        node = st.top();
                        st.pop();
                    }
                }

                if(node==-1)    continue;            
                // visited[node]=true;
                #pragma omp critical
                dfs_traversal.push_back(node);
        
                #pragma omp task firstprivate(node)
                {
                    for(auto it:graph[node]){
                        #pragma omp critical
                        {
                            if (!visited[it]) {
                                visited[it] = true;
                                st.push(it);
                            }
                        }
                    }                
                }
                #pragma omp taskwait
                        
            }
        }           
    }
}

int main(){

    for(int i=0; i<edges; i++){
        int u,v;
         u = (rand()%n);
         v = (rand()%n);
        // cin>>u>>v;

        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    //bfs
    double start = omp_get_wtime();
    for(int i=0; i<n; i++){
        if(visited[i]==false)   bfs(i);
    }
    double end = omp_get_wtime();
    // cout<<"bfs traversal : ";
    // for(auto it: bfs_traversal)cout<<it<<" ";
    cout<<" \nparallel bfs took : "<<(end-start)*1000<<" ms"<<endl;

    ///resetting the visited array
    fill(visited.begin(), visited.end(), 0);
    // cout<<"size : "<<visited.size()<<endl;

    //dfs
    start = omp_get_wtime();
    for(int i=0; i<n; i++){
        if(visited[i]==false)   dfs(i);
    }
    end= omp_get_wtime();
    // cout<<"dfs traversal : ";
    // for(auto it: dfs_traversal)cout<<it<<" ";
    cout<<"\nParallel DFS took : "<<(end-start)*1000<<" ms"<<endl;
}
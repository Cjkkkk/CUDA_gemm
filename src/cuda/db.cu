__device__ __forceinline__ void d_rank8x8( float* C, const float* A, const float* B ) { 
    float a[8], b; 
    a[0]=A[0*16]; 
    a[1]=A[1*16]; 
    a[2]=A[2*16]; 
    a[3]=A[3*16]; 
    a[4]=A[4*16]; 
    a[5]=A[5*16]; 
    a[6]=A[6*16]; 
    a[7]=A[7*16]; 
    #pragma unroll 
    for( int i=0; i<8; ++i ){ 
        b=B[i*16]; 
        C[i*8+0]+=a[0]*b; 
        C[i*8+1]+=a[1]*b; 
        C[i*8+2]+=a[2]*b; 
        C[i*8+3]+=a[3]*b; 
        C[i*8+4]+=a[4]*b; 
        C[i*8+5]+=a[5]*b; 
        C[i*8+6]+=a[6]*b; 
        C[i*8+7]+=a[7]*b; 
    } 
} 
__global__ void cuk_dgemm_unroll( 
    float* d_C, 
    const float* d_A, 
    const float* __restrict__ d_B, 
    int n, 
    int lda, 
    int ldb, 
    int ldc ) { 
    __shared__ float smem[2048]; 
    float p0, p1, p2, p3, q0, q1, q2, q3, c[64]={0.f}; 
    int k, lane, slot;
    lane=threadIdx.x&15; 
    slot=(threadIdx.y<<1)+(threadIdx.x>>4); 
    d_C+=(((blockIdx.y<<7)+slot)*ldc+(blockIdx.x<<7)+lane); 
    d_A+=(threadIdx.y*lda+(blockIdx.x<<7)+threadIdx.x); 
    d_B+=(((blockIdx.y<<7)+((threadIdx.x&1)<<4)+(threadIdx.x>>1))*ldb+threadIdx.y);
    float* St=&smem[(threadIdx.y<<7)+threadIdx.x]; 
    float* At=&smem[lane]; 
    float* Bt=&smem[1024+slot];
    if(threadIdx.y<n) { 
        p0=d_A[0*32]; 
        p1=d_A[1*32]; 
        p2=d_A[2*32]; 
        p3=d_A[3*32]; 
        q0=d_B[0*32*ldb]; 
        q1=d_B[1*32*ldb]; 
        q2=d_B[2*32*ldb]; 
        q3=d_B[3*32*ldb]; 
    } 
    for( k=n-8; k>=0; k-=8 ) { 
        *(St+0*32)=p0; 
        *(St+1*32)=p1; 
        *(St+2*32)=p2; 
        *(St+3*32)=p3; 
        *(St+0*32)=q0; 
        *(St+1*32)=q1; 
        *(St+2*32)=q2; 
        *(St+3*32)=q3; 
        __syncthreads(); 
        if(threadIdx.y<k){ 
            d_A+=(lda<<3); 
            d_B+=8; 
            p0=d_A[0*32]; 
            p1=d_A[1*32]; 
            p2=d_A[2*32]; 
            p3=d_A[3*32]; 
            q0=d_B[0*32*ldb]; 
            q1=d_B[1*32*ldb]; 
            q2=d_B[2*32*ldb]; 
            q3=d_B[3*32*ldb]; 
        } 
        #pragma unroll 
        for( int i=0; i<8; ++i ){ 
            d_rank8x8( c, &At[i*128], &Bt[i*128] ); 
        } 
        __syncthreads(); 
    } 
    if(k!=-8) { 
        *(St+0*32)=p0; 
        *(St+1*32)=p1; 
        *(St+2*32)=p2; 
        *(St+3*32)=p3; 
        *(St+0*32)=q0; 
        *(St+1*32)=q1; 
        *(St+2*32)=q2; 
        *(St+3*32)=q3; 
        __syncthreads(); 
        do{
            d_rank8x8(c, At, Bt);
            At+=128; 
            Bt+=128; 
        }while((++k)<=0);
    }
    #pragma unroll 
    for( int i=0; i<64; i+=8 ){ 
        d_C[0*16]=c[i+0]; 
        d_C[1*16]=c[i+1]; 
        d_C[2*16]=c[i+2]; 
        d_C[3*16]=c[i+3]; 
        d_C[4*16]=c[i+4]; 
        d_C[5*16]=c[i+5]; 
        d_C[6*16]=c[i+6]; 
        d_C[7*16]=c[i+7]; 
        d_C+=(ldc<<4); 
    } 
}
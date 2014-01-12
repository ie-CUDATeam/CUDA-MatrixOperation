#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define SIZE        (4096)
#define BLOCK_DIM_X (64)
#define BLOCK_DIM_Y (16)
#define BLOCK_SIZE  (BLOCK_DIM_X * BLOCK_DIM_Y)
#define GRID_DIM_X  (SIZE / BLOCK_DIM_X)
#define GRID_DIM_Y  (SIZE / BLOCK_DIM_Y)
#define GRID_SIZE   (GRID_DIM_X * GRID_DIM_Y)

void showMatrix(int *matrix);
__global__ void matrixSum(int *matrixA, int *matrixB, int *matrixC);


int main(int argc, char* argv[])
{
    const size_t matrixMemSize = sizeof(int) * SIZE * SIZE;

    // ホスト側のメモリ領域確保
    int *hostA, *hostB, *hostC;
    hostA = (int *) malloc( matrixMemSize );
    hostB = (int *) malloc( matrixMemSize );
    hostC = (int *) malloc( matrixMemSize );

    // 乱数系列の初期化
    srandom( (unsigned) time(NULL) );
    // 初期化処理
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++) {
            hostA[y * SIZE + x] = random() % 50;
            hostB[y * SIZE + x] = random() % 50;
            hostC[y * SIZE + x] = 0;
        }
    }

    // デバイス側のメモリ領域確保 & データ転送
    int *deviceA, *deviceB, *deviceC;
    cudaMalloc( (void **)&deviceA, matrixMemSize );
    cudaMalloc( (void **)&deviceB, matrixMemSize );
    cudaMalloc( (void **)&deviceC, matrixMemSize );
    cudaMemcpy( deviceA, hostA, matrixMemSize, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceB, hostB, matrixMemSize, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceC, hostC, matrixMemSize, cudaMemcpyHostToDevice );


    // グリッド & ブロックサイズの設定
    dim3 grid(GRID_DIM_X, GRID_DIM_Y);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    // 時間計測開始
    cudaEvent_t  start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // 行列和を計算
    matrixSum<<< grid, block >>>( deviceA, deviceB, deviceC );
    cudaThreadSynchronize();

    // 時間計測終了
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    // データ転送: device -> host
    cudaMemcpy( hostC, deviceC, matrixMemSize, cudaMemcpyDeviceToHost );

    // 結果表示
    // puts("matrixA =");
    // showMatrix( hostA );
    // puts("matrixB =");
    // showMatrix( hostB );
    // puts("matrixC =");
    // showMatrix( hostC );


    // 計測結果表示
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf("elapsed time: %f ms\n", elapsedTime);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );


    // デバイス側のメモリ領域解放
    cudaFree( deviceA );
    cudaFree( deviceB );
    cudaFree( deviceC );

    // ホスト側のメモリ領域解放
    free( hostA );
    free( hostB );
    free( hostC );

    return 0;
}


void showMatrix(int *matrix)
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5d ", matrix[i * SIZE + j]);
        }
        puts("");
    }
}


__global__ 
void matrixSum(int *matrixA, int *matrixB, int *matrixC)
{
    unsigned int  jx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int  jy = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int tid = (jy * SIZE) + jx;

#ifdef _USE_SHARED_MEM
    // SharedMemory を使う場合:
    unsigned int  tx = threadIdx.x;
    unsigned int  ty = threadIdx.y;

    __shared__ int sharedMatA[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ int sharedMatB[BLOCK_DIM_Y][BLOCK_DIM_X];

    sharedMatA[ty][tx] = matrixA[tid];
    sharedMatB[ty][tx] = matrixB[tid];
    __syncthreads();

    matrixC[tid] = sharedMatA[ty][tx] + sharedMatB[ty][tx];
#else
    // SharedMemory を使わない場合:
    matrixC[tid] = matrixA[tid] + matrixB[tid];
#endif
}

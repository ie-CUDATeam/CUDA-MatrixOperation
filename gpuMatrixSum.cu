#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define SIZE       32
#define BLOCK_SIZE  1

void showMatrix(int *matrix);
__global__ void matrixSum(int *matrixA, int *matrixB, int *matrixC);


int main(int argc, char* argv[])
{
    // 時間計測開始
    cudaEvent_t  start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );


    const size_t matrixSize = sizeof(int) * SIZE * SIZE;

    // ホスト側のメモリ領域確保
    int *hostA, *hostB, *hostC;
    hostA = (int *) malloc( matrixSize );
    hostB = (int *) malloc( matrixSize );
    hostC = (int *) malloc( matrixSize );

    // 乱数系列の初期化
    srandom( (unsigned) time(NULL) );
    // 初期化処理
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            hostA[i * SIZE + j] = random() % 50;
            hostB[i * SIZE + j] = random() % 50;
            hostC[i * SIZE + j] = 0;
        }
    }

    // デバイス側のメモリ領域確保 & データ転送
    int *deviceA, *deviceB, *deviceC;
    cudaMalloc( (void **)&deviceA, matrixSize );
    cudaMalloc( (void **)&deviceB, matrixSize );
    cudaMalloc( (void **)&deviceC, matrixSize );
    cudaMemcpy( deviceA, hostA, matrixSize, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceB, hostB, matrixSize, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceC, hostC, matrixSize, cudaMemcpyHostToDevice );


    // グリッド & ブロックサイズの設定
    dim3 grid(BLOCK_SIZE, BLOCK_SIZE);
    dim3 block(SIZE/BLOCK_SIZE, SIZE/BLOCK_SIZE);
    // 行列和を計算
    matrixSum<<< grid, block >>>( deviceA, deviceB, deviceC );
    // データ転送: device -> host
    cudaMemcpy( hostC, deviceC, matrixSize, cudaMemcpyDeviceToHost );


    // 結果表示
    // puts("matrixA =");
    // showMatrix( hostA );
    // puts("matrixB =");
    // showMatrix( hostB );
    // puts("matrixC =");
    // showMatrix( hostC );


    // 時間計測終了
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

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
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    matrixC[(y * SIZE) + x] 
        = matrixA[(y * SIZE) + x] + matrixB[(y * SIZE) + x];
}

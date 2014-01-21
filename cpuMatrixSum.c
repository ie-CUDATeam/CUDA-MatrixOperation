#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define SIZE 1024

void showMatrix(int *matrix);
double gettimeofday_msec();


int main(int argc, char* argv[])
{
    int *matrixA, *matrixB, *matrixC;

    /* メモリ領域確保 */
    matrixA = (int *) malloc( sizeof(int) * SIZE * SIZE );
    matrixB = (int *) malloc( sizeof(int) * SIZE * SIZE );
    matrixC = (int *) malloc( sizeof(int) * SIZE * SIZE );

    /* 乱数系列の初期化 */
    srandom( (unsigned) time(NULL) );

    /* 初期化処理 */
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrixA[i * SIZE + j] = random() % 500;
            matrixB[i * SIZE + j] = random() % 500;
            matrixC[i * SIZE + j] = 0;
        }
    }

    /* 時間計測開始 */
    double start, end;
    start = gettimeofday_msec();

    /* 行列和を計算 */
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrixC[i * SIZE + j] 
                = matrixA[i * SIZE + j] + matrixB[i * SIZE + j];
        }
    }

    /* 時間計測終了 */
    end = gettimeofday_msec();
    /* 計測結果表示 */
    printf("elapsed time: %f ms\n", end - start);

    /* 結果表示 */
    /* puts("matrixA ="); */
    /* showMatrix( matrixA ); */
    /* puts("matrixB ="); */
    /* showMatrix( matrixB ); */
    /* puts("matrixC ="); */
    /* showMatrix( matrixC ); */


    /* メモリ領域解放 */
    free( matrixA );
    free( matrixB );
    free( matrixC );

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

double gettimeofday_msec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1e+3) + (tv.tv_usec * 1e-3);
}

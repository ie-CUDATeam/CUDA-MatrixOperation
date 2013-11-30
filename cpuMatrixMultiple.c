#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 16

void showMatrix(int *matrix);


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
            matrixA[i * SIZE + j] = random() % 50;
            matrixB[i * SIZE + j] = random() % 50;
            matrixC[i * SIZE + j] = 0;
        }
    }

    /* 行列積を計算 */
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                matrixC[i * SIZE + j] 
                    += matrixA[i * SIZE + k] * matrixB[k * SIZE + j];
            }
        }
    }

    /* 結果表示 */
    puts("matrixA =");
    showMatrix( matrixA );
    puts("matrixB =");
    showMatrix( matrixB );
    puts("matrixC =");
    showMatrix( matrixC );


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

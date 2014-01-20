#ifdef _SIZE
#define SIZE (_SIZE)
#else
#define SIZE (1024)
#endif

#if defined( _BLOCK_DIM_X ) && defined( _BLOCK_DIM_Y )
#define BLOCK_DIM_X (_BLOCK_DIM_X)
#define BLOCK_DIM_Y (_BLOCK_DIM_Y)
#else
#define BLOCK_DIM_X (32)
#define BLOCK_DIM_Y (32)
#endif


#define BLOCK_SIZE  (BLOCK_DIM_X * BLOCK_DIM_Y)
#define GRID_DIM_X  (SIZE / BLOCK_DIM_X)
#define GRID_DIM_Y  (SIZE / BLOCK_DIM_Y)
#define GRID_SIZE   (GRID_DIM_X * GRID_DIM_Y)

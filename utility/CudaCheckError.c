#define CHECK_ERROR(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); EXIT_FAILURE;}}

/* if you want write it in multiple line you must add a "\" character at the end of every line as the follow example
 *
 * #define CHECK_ERROR(call) { \
 *  cudaError_t err = call; \
 *  if (err != cudaSuccess) { \
 *		  printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); EXIT_FAILURE; \
 *	  } \
 * }
 */
 
 // You can now use the macro as follow:
 CHECK_ERROR(cudaMalloc((void**)&d_A, size));

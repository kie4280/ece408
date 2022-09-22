#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define BIG_TILE 10
#define KERNEL_WIDTH 3
//@@ Define constant memory for device kernel here

__constant__ float Mask[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float M[BIG_TILE][BIG_TILE][BIG_TILE];
  int glob_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int glob_y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int glob_z = blockIdx.z * TILE_WIDTH + threadIdx.z;
  int tile_x = glob_x - KERNEL_WIDTH / 2;
  int tile_y = glob_y - KERNEL_WIDTH / 2;
  int tile_z = glob_z - KERNEL_WIDTH / 2;
  if (tile_x < x_size && tile_x >= 0 && tile_y < y_size && tile_y >= 0 && tile_z < z_size && tile_z >= 0) {
    M[threadIdx.z][threadIdx.y][threadIdx.x] = input[tile_z * y_size * x_size + tile_y * x_size + tile_x];
  } else {
    M[threadIdx.z][threadIdx.y][threadIdx.x] = 0; 
  }
  __syncthreads(); 
  float P = 0;
  if (threadIdx.z < TILE_WIDTH && threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {

    for (int a=0; a<KERNEL_WIDTH; ++a) {
      for (int b=0; b<KERNEL_WIDTH; ++b) {
        for (int c=0; c<KERNEL_WIDTH; ++c) {
          P += M[threadIdx.z + a][threadIdx.y + b][threadIdx.x + c] * Mask[a][b][c];
        }
      }
    }
    if (glob_x < x_size && glob_y < y_size && glob_z < z_size) {
      output[glob_z * y_size * x_size + glob_y * x_size + glob_x] = P;
    }
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27); // the kernel is 3x3x3

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions

  cudaMalloc((void **) &deviceInput, sizeof(float) * (inputLength - 3));
  cudaMalloc((void **) &deviceOutput, sizeof(float) * (inputLength - 3));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu

  cudaMemcpy(deviceInput, hostInput + 3, sizeof(float) * (inputLength - 3), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mask, hostKernel, sizeof(float) * kernelLength, 0, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  int threads = KERNEL_WIDTH + TILE_WIDTH - 1;
  dim3 grid(ceil(x_size * 1.0 / TILE_WIDTH), ceil(y_size * 1.0 / TILE_WIDTH), ceil(z_size * 1.0 / TILE_WIDTH));
  dim3 block(threads, threads, threads);

  //@@ Launch the GPU kernel here

  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)


  cudaMemcpy(hostOutput + 3, deviceOutput, sizeof(float) * (inputLength - 3), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

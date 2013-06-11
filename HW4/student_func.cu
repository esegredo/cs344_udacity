//Udacity HW 4
//Radix Sorting

#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histogram(unsigned int *d_bins, const unsigned int * const d_in, const unsigned int mask, const unsigned int bitPos, const size_t numElems) {
  
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= numElems)
    return;

  unsigned int myItem = d_in[myId];
  unsigned int myBin = (myItem & mask) >> bitPos;
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void scan(unsigned int *d_out, unsigned int *d_sums, const unsigned int * const d_in, const unsigned int numBins, const unsigned int numElems)  {

  extern __shared__ unsigned int sdata[];
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int offset = 1;

  // load two items per thread into shared memory
  if ((2 * myId) < numBins) {
    sdata[2 * tid] = d_in[2 * myId];
  }
  else {
    sdata[2 * tid] = 0;
  }
  
  if ((2 * myId + 1) < numBins) {
    sdata[2 * tid + 1] = d_in[2 * myId + 1];
  }
  else {
    sdata[2 * tid + 1] = 0;
  }

 	// Reduce
  for (unsigned int d = numElems >> 1; d > 0; d >>= 1) {
    if (tid < d)  {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      sdata[bi] += sdata[ai];
    }
    offset *= 2;
    __syncthreads();
  }
    
  // clear the last element
  if (tid == 0) {
    d_sums[blockIdx.x] = sdata[numElems - 1];
    sdata[numElems - 1] = 0;
  }
  
  // Down Sweep
  for (unsigned int d = 1; d < numElems; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
 	    unsigned int t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] += t;
    }
    __syncthreads();
  }
 
  // write the output to global memory
  if ((2 * myId) < numBins) {
    d_out[2 * myId] = sdata[2 * tid];
  }
  if ((2 * myId + 1) < numBins) {
    d_out[2 * myId + 1] = sdata[2 * tid + 1];
  }
}

// This version only works for one single block!
__global__ void scan2(unsigned int *d_out, const unsigned int * const d_in, const unsigned int numBins, const unsigned int numElems)  {

  extern __shared__ unsigned int sdata[];
  int tid = threadIdx.x;
  int offset = 1;

  // load two items per thread into shared memory
  if ((2 * tid) < numBins) {
    sdata[2 * tid] = d_in[2 * tid];  
  }
  else {
    sdata[2 * tid] = 0;
  }

  if ((2 * tid + 1) < numBins) {
    sdata[2 * tid + 1] = d_in[2 * tid + 1];  
  }
  else {
    sdata[2 * tid + 1] = 0;
  }

 	// Reduce
  for (unsigned int d = numElems >> 1; d > 0; d >>= 1) {
    if (tid < d)  {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      sdata[bi] += sdata[ai];
    }
    offset *= 2;
    __syncthreads();
  }
    
  // clear the last element
  if (tid == 0) {
    sdata[numElems - 1] = 0;
  }
  
  // Down Sweep
  for (unsigned int d = 1; d < numElems; d *= 2) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
 	    unsigned int t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] += t;
    }
    __syncthreads();
  }
 
  // write the output to global memory
  if ((2 * tid) < numBins) {
    d_out[2 * tid] = sdata[2 * tid];
  }

  if ((2 * tid + 1) < numBins) {
    d_out[2 * tid + 1] = sdata[2 * tid + 1];
  }
}

__global__ void add_scan(unsigned int *d_out, const unsigned int * const d_in, const unsigned int numBins) {

  if (blockIdx.x == 0)
    return;

  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int myOffset = d_in[blockIdx.x];

  if ((2 * myId) < numBins) {
    d_out[2 * myId] += myOffset;
  }
  if ((2 * myId + 1) < numBins) {
    d_out[2 * myId + 1] += myOffset;
  }

}

__global__ void pred_map(unsigned int * d_out, const unsigned int * const d_in, const unsigned int mask, const unsigned int bitPos, const unsigned int currentBin, const unsigned int numElems) {

  int myId = blockIdx.x * blockDim.x + threadIdx.x;

  if (myId >= numElems)
    return;

  unsigned int myBin = (d_in[myId] & mask) >> bitPos;

  if (myBin == currentBin)
    d_out[myId] = 1;
}

__global__ void scatter(unsigned int * d_vals_dst, unsigned int * d_pos_dst, const unsigned int * const d_vals_src, const unsigned int * const d_pos_src, const unsigned int * const d_binScan, const unsigned int * const d_inter, const unsigned int * const d_pred, const unsigned int currentBin, const unsigned int numElems) {

  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= numElems)
    return;

  if (d_pred[myId] == 0)
    return;

  unsigned int myAddress = d_inter[myId] + d_binScan[currentBin];
  d_vals_dst[myAddress] = d_vals_src[myId];
  d_pos_dst[myAddress] = d_pos_src[myId];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

  const int numBits = 4;
  const int numBins = 1 << numBits;
  
  unsigned int *d_binHistogram;
  unsigned int *d_binScan; 
  unsigned int *d_pred;
  unsigned int *d_inter;
  unsigned int *d_sums;
  unsigned int *d_incr;

  unsigned int elemsToScan = 512;
  unsigned int gridSizeScan = numElems / elemsToScan;
  if (numElems % elemsToScan != 0)
    gridSizeScan++;

  checkCudaErrors(cudaMalloc(&d_binHistogram, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMalloc(&d_binScan, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMalloc(&d_pred, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_inter, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int) * gridSizeScan));
  checkCudaErrors(cudaMalloc(&d_incr, sizeof(unsigned int) * gridSizeScan));

  unsigned int *d_vals_src = d_inputVals;
  unsigned int *d_pos_src  = d_inputPos;

  unsigned int *d_vals_dst = d_outputVals;
  unsigned int *d_pos_dst  = d_outputPos;

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    checkCudaErrors(cudaMemset(d_binHistogram, 0, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMemset(d_binScan, 0, sizeof(unsigned int) * numBins));

    // Step 1: Calculate the histogram of the number of occurrences of each digit
    dim3 blockSize(256, 1, 1);
    unsigned int numBlocks = numElems / blockSize.x;
    if (numElems % blockSize.x != 0)
      numBlocks++;
    dim3 gridSize(numBlocks, 1, 1);
    histogram<<<gridSize, blockSize>>>(d_binHistogram, d_vals_src, mask, i, numElems);
  
    // Step 2: Exclusive prefix sum of the histogram
    gridSize.x = 1;
    blockSize.x = numBins / 2;
    scan2<<<gridSize, blockSize, sizeof(unsigned int) * numBins>>>(d_binScan, d_binHistogram, numBins, numBins);

    // Step 3: Determine relative offset: Compact the input taking different predicates
    for (unsigned int j = 0; j < numBins; j++) {

      // Calculate the predicate
      blockSize.x = 256;
      numBlocks = numElems / blockSize.x;
      if (numElems % blockSize.x != 0)
        numBlocks++;
      gridSize.x = numBlocks;
    
      checkCudaErrors(cudaMemset(d_pred, 0, sizeof(unsigned int) * numElems));
      pred_map<<<gridSize, blockSize>>>(d_pred, d_vals_src, mask, i, j, numElems);

      // Perform Blelloch scan to obtain relative offsets
      // First-level scan to obtain the scanned blocks
      elemsToScan = 512;
      blockSize.x = elemsToScan / 2;
      gridSize.x = numElems / elemsToScan;
      if (numElems % elemsToScan != 0)
        gridSize.x++;

      checkCudaErrors(cudaMemset(d_sums, 0, sizeof(unsigned int) * gridSize.x));
      checkCudaErrors(cudaMemset(d_inter, 0, sizeof(unsigned int) * numElems));
      scan<<<gridSize, blockSize, sizeof(unsigned int) * elemsToScan>>>(d_inter, d_sums, d_pred, numElems, elemsToScan);
  
      // Second-level scan to obtain the scanned blocks sums
      elemsToScan = gridSize.x;

      // Look for the next power of 2 (32 bits)
      unsigned int nextPow = elemsToScan;
      nextPow--;
      nextPow = (nextPow >> 1) | nextPow;
      nextPow = (nextPow >> 2) | nextPow;
      nextPow = (nextPow >> 4) | nextPow;
      nextPow = (nextPow >> 8) | nextPow;
      nextPow = (nextPow >> 16) | nextPow;
      nextPow++;

      blockSize.x = nextPow / 2;
      gridSize.x = 1;
      checkCudaErrors(cudaMemset(d_incr, 0, sizeof(unsigned int) * elemsToScan));
      scan2<<<gridSize, blockSize, sizeof(unsigned int) * nextPow>>>(d_incr, d_sums, elemsToScan, nextPow);
      
      // Add scanned block sum i to all values of scanned block i
      elemsToScan = 512;
      blockSize.x = elemsToScan / 2;
      gridSize.x = numElems / elemsToScan;
      if (numElems % elemsToScan != 0)
        gridSize.x++;
      add_scan<<<gridSize, blockSize>>>(d_inter, d_incr, numElems);

      // Scatter the results
      blockSize.x = 256;
      gridSize.x = numElems / blockSize.x;
      if (numElems % blockSize.x != 0)
        gridSize.x++;
      scatter<<<gridSize, blockSize>>>(d_vals_dst, d_pos_dst, d_vals_src, d_pos_src, d_binScan, d_inter, d_pred, j, numElems);
    }
    
    // Step 4: Swap the pointers for the next iteration
    std::swap(d_vals_dst, d_vals_src);
    std::swap(d_pos_dst, d_pos_src);
  }
  
  // Copy from input to output
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  
  // Free the memory
  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_binScan));
  checkCudaErrors(cudaFree(d_pred));
  checkCudaErrors(cudaFree(d_inter));
  checkCudaErrors(cudaFree(d_sums));
  checkCudaErrors(cudaFree(d_incr));
}

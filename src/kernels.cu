#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/** @brief 计算矩阵 trace (对角线元素之和) 的 CUDA kernel
 *
 * 该 kernel 输入矩阵以行主序存储在一维数组中, 最终结果存储在 output 指针指向的位置.
 * 每个线程读取一个对角线元素, 然后在 block 内使用 shared memory 进行归约
 *
 * @tparam T 矩阵元素的数据类型 (如 float, int)
 * @param input 输入矩阵在设备内存中的指针 (行主序展开).
 * @param output 输出标量在设备内存中的指针.
 * @param diag_len 对角线长度 (min(rows, cols)).
 * @param cols 矩阵的列数.
 */

template <typename T>
__global__ void trace_kernel(const T* input, T* output, size_t diag_len, size_t cols) {
    extern __shared__ char shared_mem[];
    // 将共享内存转换为 T* 类型
    T* sdata = reinterpret_cast<T*>(shared_mem);

    unsigned int idx = threadIdx.x; // 线程在 block 内的索引, 用于访问共享内存
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引

    // 每个线程读取一个对角线元素：input[tid * cols + tid];
    // 对角线元素索引为 (tid, tid), 其在行主序存储中位置为 tid * cols + tid.
    sdata[idx] = (tid < diag_len) ? input[tid * cols + tid] : T(0);
    __syncthreads();

    // 树形归约, 每次迭代将活跃线程数减半
    // s 从 blockDim.x/2 开始，每次除以 2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
        sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    // block 内第一个线程将结果原子加到全局输出
    if (idx == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // 对角线长度为 min(rows, cols)
    size_t diag_len = (rows < cols) ? rows : cols;
    if (diag_len == 0) { return T(0); }

    // 分配设备内存
    T* d_input;
    T* d_output;
    size_t input_size = rows * cols * sizeof(T);

    RUNTIME_CHECK(cudaMalloc(&d_input, input_size));
    RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));

    // 拷贝输入数据到设备内存
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

    // 初始化输出为 0
    T zero = T(0);
    RUNTIME_CHECK(cudaMemcpy(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    // 每个 Block 使用 256 个线程, 根据对角线长度计算需要的 Block 数量
    const int num_threads = 256;
    int num_blocks = (diag_len + num_threads - 1) / num_threads;
    size_t shared_mem_size = num_threads * sizeof(T);

    // 启动 kernel
    trace_kernel<T><<<num_blocks, num_threads, shared_mem_size>>>(
        d_input, d_output, diag_len, cols);

    // 检查 kernel 执行错误
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回 host
    T result;
    RUNTIME_CHECK(cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost));

    // 释放设备内存
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

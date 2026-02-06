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

// 将任意类型转换为 float 进行计算
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

// 将 float 转换到目标类型
__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ half from_float(float x, half*) { return __float2half(x); }

template <typename T, int HEAD_DIM_MAX = 128>
__global__ void attention_kernel(
    const T* __restrict__ Q,    // [batch, tgt_len, q_heads, head_dim]
    const T* __restrict__ K,    // [batch, src_len, kv_heads, head_dim]
    const T* __restrict__ V,    // [batch, src_len, kv_heads, head_dim]
    T* __restrict__ O,          // [batch, tgt_len, q_heads, head_dim]
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    float scale, bool is_causal)
{
    // 计算当前线程块负责的 batch, head, query position
    int batch_head = blockIdx.x;
    int batch = batch_head / query_heads;
    int head = batch_head % query_heads;
    int q_pos = blockIdx.y;
  
    // GQA: 计算对应的 KV head
    // query_heads 是 kv_heads 的整数倍，多个 query heads 共享一个 kv head
    int heads_per_kv = query_heads / kv_heads;
    int kv_head = head / heads_per_kv;
    
    // 边界检查
    if (batch >= batch_size || q_pos >= target_seq_len) return;
    
    int tid = threadIdx.x;
  
    // 计算输入输出的基地址
    // Q/O: [batch, seq, heads, dim] -> batch * (tgt_len * q_heads * dim) + seq * (q_heads * dim) + head * dim
    // K/V: [batch, seq, heads, dim] -> batch * (src_len * kv_heads * dim) + seq * (kv_heads * dim) + head * dim
    int q_base = batch * (target_seq_len * query_heads * head_dim) 
                + q_pos * (query_heads * head_dim) 
                + head * head_dim;
    int kv_batch_base = batch * (src_seq_len * kv_heads * head_dim);
    
    // 每个线程加载 Q 的一部分到寄存器
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = to_float(Q[q_base + tid]);
    }
  
    // 使用 shared memory 存储 K, V 的一个位置的数据
    extern __shared__ char smem[];
    float* s_kv = reinterpret_cast<float*>(smem);  // [2 * head_dim] for K and V
    float* s_k = s_kv;
    float* s_v = s_kv + head_dim;
  
    // Online softmax
    float m = -INFINITY;
    float l = 0.0f;
    float o_acc = 0.0f;  // 每个线程负责输出的一个维度
    
    // Causal mask 的上界：query position q_pos 只能 attend 到 <= q_pos 的位置
    int kv_end = is_causal ? min(q_pos + 1, src_seq_len) : src_seq_len;
    
    // 遍历所有 KV positions
    for (int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        // 计算当前 KV position 的基地址
        int kv_base = kv_batch_base + kv_pos * (kv_heads * head_dim) + kv_head * head_dim;
        
        // 协作加载 K[kv_pos] 和 V[kv_pos] 到 shared memory
        if (tid < head_dim) {
            s_k[tid] = to_float(K[kv_base + tid]);
            s_v[tid] = to_float(V[kv_base + tid]);
        }
        __syncthreads();
        
        // 计算 attention score: Q[q_pos] · K[kv_pos] * scale
        float score = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float q_d = (d < head_dim && d == tid) ? q_val : 
                        (d < head_dim ? to_float(Q[q_base + d]) : 0.0f);
            score += q_d * s_k[d];
        }
        
        // 更简单的方式：让每个线程计算完整的点积
        score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += to_float(Q[q_base + d]) * s_k[d];
        }
        score *= scale;
        
        // Online softmax 更新
        // 计算新的 max
        float m_new = fmaxf(m, score);
        
        // 计算校正因子
        // alpha: 用于缩放之前累加的值
        // beta: 用于缩放当前的值
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);
        
        // 更新 running sum
        l = l * alpha + beta;
        
        // 更新 output accumulator
        // o_new = o_old * alpha + exp(score - m_new) * V[kv_pos]
        if (tid < head_dim) {
            o_acc = o_acc * alpha + beta * s_v[tid];
        }
        
        // 更新 running max
        m = m_new;
        
        __syncthreads();
    }
    
    // 最终归一化并写回输出
    // O[q_pos] = o_acc / l
    if (tid < head_dim && l > 0.0f) {
        int o_idx = q_base + tid;
        T* dummy = nullptr;
        O[o_idx] = from_float(o_acc / l, dummy);
    }
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
    // 计算各张量大小
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;
    
    // 确保输出向量大小正确
    h_o.resize(o_size);
    
    // 分配设备内存
    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    // 拷贝输入到设备
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // 计算 scale = 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // 配置 kernel 启动参数
    // Grid: (batch_size * query_heads, target_seq_len)
    // Block: head_dim 线程（或最少 32 线程以利用 warp）
    dim3 grid(batch_size * query_heads, target_seq_len);
    int block_size = std::max(32, head_dim);  // 至少 32 线程以保证 warp 效率
    
    // Shared memory: 存储 K 和 V 的一个位置的数据
    size_t smem_size = 2 * head_dim * sizeof(float);
    
    // 启动 kernel
    attention_kernel<T><<<grid, block_size, smem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal);
    
    // 检查 kernel 执行错误
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回 host
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
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

/*
 * VE Device Kernels for Binary/Ternary Neural Network Operations
 * 
 * These kernels run on the NEC SX-Aurora TSUBASA Vector Engine.
 * They are compiled with ncc and loaded via VEDA from the x86 host.
 *
 * Build: ncc -O4 -fpic -shared -o libve_kernels.so ve_kernels.c
 */

#include <stdint.h>
#include <stddef.h>

/* VE popcount - ncc supports __builtin_popcountll */
static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

/*
 * Binary dot product for 256 bits (4 x uint64_t)
 * Returns XNOR popcount (0 to 256)
 */
uint64_t ve_binary_dot_256(const uint64_t* weights, const uint64_t* acts) {
    uint64_t count = 0;
    
    /* Unrolled for 256 bits = 4 x 64-bit words */
    uint64_t xnor0 = ~(weights[0] ^ acts[0]);
    uint64_t xnor1 = ~(weights[1] ^ acts[1]);
    uint64_t xnor2 = ~(weights[2] ^ acts[2]);
    uint64_t xnor3 = ~(weights[3] ^ acts[3]);
    
    count += popcount64(xnor0);
    count += popcount64(xnor1);
    count += popcount64(xnor2);
    count += popcount64(xnor3);
    
    return count;
}

/*
 * Binary dot product for n bits (multiple of 256)
 * Returns total XNOR popcount
 */
uint64_t ve_binary_dot_n(const uint64_t* weights, 
                          const uint64_t* acts, 
                          uint64_t n_bits) {
    uint64_t total = 0;
    uint64_t n_words = n_bits / 64;
    
    /* Process 4 words (256 bits) at a time */
    for (uint64_t i = 0; i < n_words; i += 4) {
        uint64_t xnor0 = ~(weights[i+0] ^ acts[i+0]);
        uint64_t xnor1 = ~(weights[i+1] ^ acts[i+1]);
        uint64_t xnor2 = ~(weights[i+2] ^ acts[i+2]);
        uint64_t xnor3 = ~(weights[i+3] ^ acts[i+3]);
        
        total += popcount64(xnor0);
        total += popcount64(xnor1);
        total += popcount64(xnor2);
        total += popcount64(xnor3);
    }
    
    return total;
}

/*
 * Ternary dot product for 256 elements
 * Ternary encoding: (sign_bit, nonzero_bit)
 *   - (1, 1) = +1
 *   - (0, 1) = -1
 *   - (*, 0) = 0
 * Returns: positive_count - negative_count
 */
int64_t ve_ternary_dot_256(const uint64_t* w_sign, const uint64_t* w_nonzero,
                           const uint64_t* a_sign, const uint64_t* a_nonzero) {
    int64_t positive = 0;
    int64_t negative = 0;
    
    for (int i = 0; i < 4; i++) {
        uint64_t both_nz = w_nonzero[i] & a_nonzero[i];
        uint64_t same_sign = ~(w_sign[i] ^ a_sign[i]);
        
        uint64_t pos = both_nz & same_sign;
        uint64_t neg = both_nz & ~same_sign;
        
        positive += popcount64(pos);
        negative += popcount64(neg);
    }
    
    return positive - negative;
}

/*
 * Ternary dot product for n elements (multiple of 256)
 */
int64_t ve_ternary_dot_n(const uint64_t* w_sign, const uint64_t* w_nonzero,
                         const uint64_t* a_sign, const uint64_t* a_nonzero,
                         uint64_t n_elements) {
    int64_t total = 0;
    uint64_t n_words = n_elements / 64;
    
    for (uint64_t i = 0; i < n_words; i += 4) {
        int64_t positive = 0;
        int64_t negative = 0;
        
        for (int j = 0; j < 4; j++) {
            uint64_t both_nz = w_nonzero[i+j] & a_nonzero[i+j];
            uint64_t same_sign = ~(w_sign[i+j] ^ a_sign[i+j]);
            
            uint64_t pos = both_nz & same_sign;
            uint64_t neg = both_nz & ~same_sign;
            
            positive += popcount64(pos);
            negative += popcount64(neg);
        }
        
        total += positive - negative;
    }
    
    return total;
}

/*
 * Binary matrix-vector multiply: y = W * x
 * W is (m x n) binary matrix (packed bits), x is (n,) binary vector
 * Output y[i] = 2 * popcount(XNOR(W[i], x)) - n (in {-1,+1} domain)
 * 
 * This version is parallelized across rows using OpenMP.
 */
void ve_binary_matvec(int64_t* y,
                      const uint64_t* W,
                      const uint64_t* x,
                      uint64_t m,
                      uint64_t n) {
    uint64_t words_per_row = n / 64;
    
    #pragma omp parallel for
    for (uint64_t row = 0; row < m; row++) {
        const uint64_t* W_row = W + row * words_per_row;
        uint64_t sum = 0;
        
        for (uint64_t i = 0; i < words_per_row; i += 4) {
            uint64_t xnor0 = ~(W_row[i+0] ^ x[i+0]);
            uint64_t xnor1 = ~(W_row[i+1] ^ x[i+1]);
            uint64_t xnor2 = ~(W_row[i+2] ^ x[i+2]);
            uint64_t xnor3 = ~(W_row[i+3] ^ x[i+3]);
            
            sum += popcount64(xnor0);
            sum += popcount64(xnor1);
            sum += popcount64(xnor2);
            sum += popcount64(xnor3);
        }
        
        /* Convert to {-1, +1} domain */
        y[row] = 2 * (int64_t)sum - (int64_t)n;
    }
}

/*
 * Ternary matrix-vector multiply: y = W * x
 * W and x are ternary encoded (sign + nonzero bits)
 */
void ve_ternary_matvec(int64_t* y,
                       const uint64_t* W_sign,
                       const uint64_t* W_nonzero,
                       const uint64_t* x_sign,
                       const uint64_t* x_nonzero,
                       uint64_t m,
                       uint64_t n) {
    uint64_t words_per_row = n / 64;
    
    #pragma omp parallel for
    for (uint64_t row = 0; row < m; row++) {
        const uint64_t* Ws = W_sign + row * words_per_row;
        const uint64_t* Wn = W_nonzero + row * words_per_row;
        int64_t total = 0;
        
        for (uint64_t i = 0; i < words_per_row; i += 4) {
            int64_t positive = 0;
            int64_t negative = 0;
            
            for (int j = 0; j < 4; j++) {
                uint64_t both_nz = Wn[i+j] & x_nonzero[i+j];
                uint64_t same_sign = ~(Ws[i+j] ^ x_sign[i+j]);
                
                uint64_t pos = both_nz & same_sign;
                uint64_t neg = both_nz & ~same_sign;
                
                positive += popcount64(pos);
                negative += popcount64(neg);
            }
            
            total += positive - negative;
        }
        
        y[row] = total;
    }
}

/*
 * Batched binary matrix-vector multiply for multiple inputs
 * y[b] = W * x[b] for b in [0, batch_size)
 */
void ve_binary_matvec_batch(int64_t* y,
                            const uint64_t* W,
                            const uint64_t* x,
                            uint64_t m,
                            uint64_t n,
                            uint64_t batch_size) {
    uint64_t words_per_row = n / 64;
    uint64_t x_stride = n / 64;
    
    #pragma omp parallel for collapse(2)
    for (uint64_t b = 0; b < batch_size; b++) {
        for (uint64_t row = 0; row < m; row++) {
            const uint64_t* W_row = W + row * words_per_row;
            const uint64_t* x_b = x + b * x_stride;
            uint64_t sum = 0;
            
            for (uint64_t i = 0; i < words_per_row; i += 4) {
                uint64_t xnor0 = ~(W_row[i+0] ^ x_b[i+0]);
                uint64_t xnor1 = ~(W_row[i+1] ^ x_b[i+1]);
                uint64_t xnor2 = ~(W_row[i+2] ^ x_b[i+2]);
                uint64_t xnor3 = ~(W_row[i+3] ^ x_b[i+3]);
                
                sum += popcount64(xnor0);
                sum += popcount64(xnor1);
                sum += popcount64(xnor2);
                sum += popcount64(xnor3);
            }
            
            y[b * m + row] = 2 * (int64_t)sum - (int64_t)n;
        }
    }
}

/*
 * Scaled ternary matrix-vector multiply with float output
 * y[row] = scale * sum(W[row][i] * x[i])
 */
void ve_ternary_matvec_scaled(float* y,
                              const uint64_t* W_sign,
                              const uint64_t* W_nonzero,
                              const uint64_t* x_sign,
                              const uint64_t* x_nonzero,
                              float scale,
                              uint64_t m,
                              uint64_t n) {
    uint64_t words_per_row = n / 64;
    
    #pragma omp parallel for
    for (uint64_t row = 0; row < m; row++) {
        const uint64_t* Ws = W_sign + row * words_per_row;
        const uint64_t* Wn = W_nonzero + row * words_per_row;
        int64_t total = 0;
        
        for (uint64_t i = 0; i < words_per_row; i += 4) {
            int64_t positive = 0;
            int64_t negative = 0;
            
            for (int j = 0; j < 4; j++) {
                uint64_t both_nz = Wn[i+j] & x_nonzero[i+j];
                uint64_t same_sign = ~(Ws[i+j] ^ x_sign[i+j]);
                
                uint64_t pos = both_nz & same_sign;
                uint64_t neg = both_nz & ~same_sign;
                
                positive += popcount64(pos);
                negative += popcount64(neg);
            }
            
            total += positive - negative;
        }
        
        y[row] = scale * (float)total;
    }
}

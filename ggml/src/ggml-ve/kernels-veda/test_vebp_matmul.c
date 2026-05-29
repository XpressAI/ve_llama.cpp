/* Standalone correctness test for ve_vebp_matmul_ptr_inner (batched VEBP matmul)
 * vs the per-column ve_vebp_matvec_ptr_inner loop the graph compiler used.
 * Random interleaved planes are fine — both kernels run the SAME vpcnt math, so
 * the batched output must match the per-column reference column-for-column.
 *
 * Build: ncc -O4 -fopenmp test_vebp_matmul.c libve_sgemv.so -o /tmp/test_vebp_matmul
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern void ve_vebp_matvec_ptr_inner(float *y, const uint64_t *ws, const uint64_t *wn,
                                     const float *wsc, const float *x, int M, int K);
extern void ve_vebp_matmul_ptr_inner(float *y, const uint64_t *ws, const uint64_t *wn,
                                     const float *wsc, const float *x, int M, int K, int N);

/* Stubs for the .so's device-runtime / cblas / ftrace deps — none are reached
 * from the matvec/matmul _inner code paths this test exercises. */
int  vedaMemPtr(void **p, uint64_t v) { (void) p; (void) v; return 1; }
void cblas_sgemv(void) {}
void cblas_sgemm(void) {}
void __ftrace_func_enter(void) {}
void __ftrace_func_exit(void) {}

#define RB 256

static int run_case(int M, int K, int N) {
    long wpr = K / 64, ng = K / 128;
    long Mblk = (M + RB - 1) / RB;            /* padded full blocks */
    long ws_n = Mblk * wpr * RB, wsc_n = Mblk * ng * RB;
    uint64_t *ws = aligned_alloc(64, ws_n * 8), *wn = aligned_alloc(64, ws_n * 8);
    float *wsc = aligned_alloc(64, wsc_n * 4);
    float *x  = aligned_alloc(64, (long) K * N * 4);
    float *yr = aligned_alloc(64, (long) M * N * 4);
    float *yb = aligned_alloc(64, (long) M * N * 4);

    for (long i = 0; i < ws_n; i++) { ws[i] = ((uint64_t) rand() << 32) ^ rand();
                                      wn[i] = ((uint64_t) rand() << 32) ^ rand(); }
    for (long i = 0; i < wsc_n; i++) wsc[i] = (float) (rand() % 1000) / 1000.0f;
    for (long i = 0; i < (long) K * N; i++) x[i] = (float) (rand() % 2000 - 1000) / 100.0f;

    for (int j = 0; j < N; j++) {
        #pragma omp parallel
        { ve_vebp_matvec_ptr_inner(yr + (long) j * M, ws, wn, wsc, x + (long) j * K, M, K); }
    }
    #pragma omp parallel
    { ve_vebp_matmul_ptr_inner(yb, ws, wn, wsc, x, M, K, N); }

    double maxd = 0; long nbad = 0;
    for (long i = 0; i < (long) M * N; i++) {
        double d = fabs((double) yr[i] - (double) yb[i]);
        if (d > maxd) maxd = d;
        if (d > 1e-3) nbad++;
    }
    printf("M=%d K=%d N=%d  max|ref-batch|=%.3e  nbad=%ld/%ld  %s\n",
           M, K, N, maxd, nbad, (long) M * N, nbad ? "FAIL" : "PASS");
    free(ws); free(wn); free(wsc); free(x); free(yr); free(yb);
    return nbad != 0;
}

int main(void) {
    srand(12345);
    int bad = 0;
    bad |= run_case(768, 512, 5);    /* clean: M=3 blocks, no tail */
    bad |= run_case(800, 512, 5);    /* tail: 3 full blocks + 32 rows */
    bad |= run_case(4096, 1024, 8);  /* bigger, prompt-ish N */
    return bad;
}

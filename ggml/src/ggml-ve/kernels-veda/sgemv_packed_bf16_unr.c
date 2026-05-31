#include <stdio.h>
#include <velintrin.h>

#define VLEN (256)
typedef unsigned short bf16;

#define load_bf16_to_packed_fp32(wv,wp,vlen) \
    do { \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vlen); \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vlen); \
        wr = _vel_vand_vvvl(wr, bf16mskl, vlen); \
        wv = _vel_vor_vvvl(wv, wr, vlen); \
    } while(0)

#define sumup_packed_fp32_store(tv,yt,VLEN) \
    do { \
        __vr rv = _vel_vand_vvvl(tv, low32msk, VLEN); \
        __vr sv = _vel_vsll_vvsl(rv, 32, VLEN); \
        tv = _vel_vfadds_vvvl(tv, sv, VLEN); \
        tv = _vel_vfsums_vvl(tv, VLEN); \
        yt = _vel_lvss_svs(tv, 0); \
    } while(0)


/* 4-column tile: y0..y3[d] = w[d,n] @ (x0..x3)[n], one weight row UNPACKED ONCE
 * per j-chunk and FMA'd against all 4 columns. The matvec is compute-bound on
 * the BF16->packed-FP32 unpack (load_bf16_to_packed_fp32 = 4 vector ops vs 1
 * FMA), so amortizing the unpack across 4 columns is the win — cache-blocking
 * the col-loop does NOT help because each per-column matvec re-unpacks. Keeps
 * the same BF16 x-truncation as the matvec (bit-identical per column). */
#define UNPACK_W(dst, wp) do { \
        __vr _wr; \
        dst = _vel_vldunc_vssl(4, (void *)(wp), vl); \
        _wr = _vel_vsrl_vvsl(dst, 16, vl); \
        _wr = _vel_vand_vvvl(_wr, bf16mskl, vl); \
        dst = _vel_vor_vvvl(dst, _wr, vl); \
    } while (0)
#define LOADX_TRUNC(dst, xp) do { \
        if ((unsigned long)((xp)) & 0x7) { \
            dst = _vel_vldu_vssl(8, (void *)((xp)+1), vl); \
            __vr _xl = _vel_vldlzx_vssl(8, (void *)((xp)), vl); \
            dst = _vel_pvor_vvvl(dst, _xl, vl); \
        } else { dst = _vel_vld_vssl(8, (void *)((xp)), vl); } \
        dst = _vel_vand_vvvl(dst, bf16inputmsk, vl); \
    } while (0)

void sgemm4_packed_bf16(float *y0, float *y1, float *y2, float *y3,
                        const float *x0, const float *x1, const float *x2, const float *x3,
                        bf16 *w, int n, int d) {
    float zero[2] = {0.0f, 0.0f};
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffff, VLEN);
    __vr bf16inputmsk = _vel_vbrdl_vsl(0xffff0000ffff0000UL, VLEN);
    int i = 0;

    /* 4 rows x 4 cols: 16-way ILP (like the matvec's 16-row unroll) AND the
     * weight row is unpacked ONCE per row and reused across all 4 columns. */
    for (; i + 4 <= d; i += 4) {
        __vr t00 = _vel_vld_vssl(0,&zero[0],VLEN), t01 = t00, t02 = t00, t03 = t00;
        __vr t10 = t00, t11 = t00, t12 = t00, t13 = t00;
        __vr t20 = t00, t21 = t00, t22 = t00, t23 = t00;
        __vr t30 = t00, t31 = t00, t32 = t00, t33 = t00;
        bf16 *wp0 = w + (long)(i+0)*n, *wp1 = w + (long)(i+1)*n;
        bf16 *wp2 = w + (long)(i+2)*n, *wp3 = w + (long)(i+3)*n;
        for (int j = 0; j < n; j += 2 * VLEN) {
            const int vl = n - j < 2 * VLEN ? (n - j) >> 1 : VLEN;
            __vr xv0, xv1, xv2, xv3, w0, w1, w2, w3;
            LOADX_TRUNC(xv0, x0 + j); LOADX_TRUNC(xv1, x1 + j);
            LOADX_TRUNC(xv2, x2 + j); LOADX_TRUNC(xv3, x3 + j);
            UNPACK_W(w0, wp0 + j); UNPACK_W(w1, wp1 + j);
            UNPACK_W(w2, wp2 + j); UNPACK_W(w3, wp3 + j);
            t00 = _vel_pvfmad_vvvvl(t00,xv0,w0,vl); t01 = _vel_pvfmad_vvvvl(t01,xv1,w0,vl);
            t02 = _vel_pvfmad_vvvvl(t02,xv2,w0,vl); t03 = _vel_pvfmad_vvvvl(t03,xv3,w0,vl);
            t10 = _vel_pvfmad_vvvvl(t10,xv0,w1,vl); t11 = _vel_pvfmad_vvvvl(t11,xv1,w1,vl);
            t12 = _vel_pvfmad_vvvvl(t12,xv2,w1,vl); t13 = _vel_pvfmad_vvvvl(t13,xv3,w1,vl);
            t20 = _vel_pvfmad_vvvvl(t20,xv0,w2,vl); t21 = _vel_pvfmad_vvvvl(t21,xv1,w2,vl);
            t22 = _vel_pvfmad_vvvvl(t22,xv2,w2,vl); t23 = _vel_pvfmad_vvvvl(t23,xv3,w2,vl);
            t30 = _vel_pvfmad_vvvvl(t30,xv0,w3,vl); t31 = _vel_pvfmad_vvvvl(t31,xv1,w3,vl);
            t32 = _vel_pvfmad_vvvvl(t32,xv2,w3,vl); t33 = _vel_pvfmad_vvvvl(t33,xv3,w3,vl);
        }
        sumup_packed_fp32_store(t00,y0[i+0],VLEN); sumup_packed_fp32_store(t01,y1[i+0],VLEN);
        sumup_packed_fp32_store(t02,y2[i+0],VLEN); sumup_packed_fp32_store(t03,y3[i+0],VLEN);
        sumup_packed_fp32_store(t10,y0[i+1],VLEN); sumup_packed_fp32_store(t11,y1[i+1],VLEN);
        sumup_packed_fp32_store(t12,y2[i+1],VLEN); sumup_packed_fp32_store(t13,y3[i+1],VLEN);
        sumup_packed_fp32_store(t20,y0[i+2],VLEN); sumup_packed_fp32_store(t21,y1[i+2],VLEN);
        sumup_packed_fp32_store(t22,y2[i+2],VLEN); sumup_packed_fp32_store(t23,y3[i+2],VLEN);
        sumup_packed_fp32_store(t30,y0[i+3],VLEN); sumup_packed_fp32_store(t31,y1[i+3],VLEN);
        sumup_packed_fp32_store(t32,y2[i+3],VLEN); sumup_packed_fp32_store(t33,y3[i+3],VLEN);
    }
    /* row remainder (<4): one row x 4 cols */
    for (; i < d; i++) {
        __vr t0 = _vel_vld_vssl(0,&zero[0],VLEN), t1 = t0, t2 = t0, t3 = t0;
        bf16 *wp = w + (long)i*n;
        for (int j = 0; j < n; j += 2 * VLEN) {
            const int vl = n - j < 2 * VLEN ? (n - j) >> 1 : VLEN;
            __vr xv0, xv1, xv2, xv3, wv;
            LOADX_TRUNC(xv0, x0 + j); LOADX_TRUNC(xv1, x1 + j);
            LOADX_TRUNC(xv2, x2 + j); LOADX_TRUNC(xv3, x3 + j);
            UNPACK_W(wv, wp + j);
            t0 = _vel_pvfmad_vvvvl(t0,xv0,wv,vl); t1 = _vel_pvfmad_vvvvl(t1,xv1,wv,vl);
            t2 = _vel_pvfmad_vvvvl(t2,xv2,wv,vl); t3 = _vel_pvfmad_vvvvl(t3,xv3,wv,vl);
        }
        sumup_packed_fp32_store(t0,y0[i],VLEN); sumup_packed_fp32_store(t1,y1[i],VLEN);
        sumup_packed_fp32_store(t2,y2[i],VLEN); sumup_packed_fp32_store(t3,y3[i],VLEN);
    }
}

/* 8-column tile, 4 rows x 8 cols = 32-way ILP, weight unpacked once per row and
 * reused across 8 columns (HALF the unpacks of the 4-col tile). Uses ~50 of the
 * 64 vector registers. For large-N prompt eval; small N (e.g. MTP verify N=5)
 * stays on the 4-col tile since C can't exceed N. */
void sgemm8_packed_bf16(float *y0, float *y1, float *y2, float *y3,
                        float *y4, float *y5, float *y6, float *y7,
                        const float *x0, const float *x1, const float *x2, const float *x3,
                        const float *x4, const float *x5, const float *x6, const float *x7,
                        bf16 *w, int n, int d) {
    float zero[2] = {0.0f, 0.0f};
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffff, VLEN);
    __vr bf16inputmsk = _vel_vbrdl_vsl(0xffff0000ffff0000UL, VLEN);
    int i = 0;
    const float *xp[8] = {x0,x1,x2,x3,x4,x5,x6,x7};
    float *yp[8] = {y0,y1,y2,y3,y4,y5,y6,y7};
    #define FMA8(R,W) do { \
        t##R##0=_vel_pvfmad_vvvvl(t##R##0,xv0,W,vl); t##R##1=_vel_pvfmad_vvvvl(t##R##1,xv1,W,vl); \
        t##R##2=_vel_pvfmad_vvvvl(t##R##2,xv2,W,vl); t##R##3=_vel_pvfmad_vvvvl(t##R##3,xv3,W,vl); \
        t##R##4=_vel_pvfmad_vvvvl(t##R##4,xv4,W,vl); t##R##5=_vel_pvfmad_vvvvl(t##R##5,xv5,W,vl); \
        t##R##6=_vel_pvfmad_vvvvl(t##R##6,xv6,W,vl); t##R##7=_vel_pvfmad_vvvvl(t##R##7,xv7,W,vl); } while(0)
    for (; i + 4 <= d; i += 4) {
        __vr t00=_vel_vld_vssl(0,&zero[0],VLEN),t01=t00,t02=t00,t03=t00,t04=t00,t05=t00,t06=t00,t07=t00;
        __vr t10=t00,t11=t00,t12=t00,t13=t00,t14=t00,t15=t00,t16=t00,t17=t00;
        __vr t20=t00,t21=t00,t22=t00,t23=t00,t24=t00,t25=t00,t26=t00,t27=t00;
        __vr t30=t00,t31=t00,t32=t00,t33=t00,t34=t00,t35=t00,t36=t00,t37=t00;
        bf16 *wp0=w+(long)(i+0)*n,*wp1=w+(long)(i+1)*n,*wp2=w+(long)(i+2)*n,*wp3=w+(long)(i+3)*n;
        for (int j = 0; j < n; j += 2*VLEN) {
            const int vl = n - j < 2*VLEN ? (n - j) >> 1 : VLEN;
            __vr xv0,xv1,xv2,xv3,xv4,xv5,xv6,xv7,w0,w1,w2,w3;
            LOADX_TRUNC(xv0,x0+j);LOADX_TRUNC(xv1,x1+j);LOADX_TRUNC(xv2,x2+j);LOADX_TRUNC(xv3,x3+j);
            LOADX_TRUNC(xv4,x4+j);LOADX_TRUNC(xv5,x5+j);LOADX_TRUNC(xv6,x6+j);LOADX_TRUNC(xv7,x7+j);
            UNPACK_W(w0,wp0+j);FMA8(0,w0); UNPACK_W(w1,wp1+j);FMA8(1,w1);
            UNPACK_W(w2,wp2+j);FMA8(2,w2); UNPACK_W(w3,wp3+j);FMA8(3,w3);
        }
        #define ST(R) do { sumup_packed_fp32_store(t##R##0,y0[i+R],VLEN);sumup_packed_fp32_store(t##R##1,y1[i+R],VLEN); \
            sumup_packed_fp32_store(t##R##2,y2[i+R],VLEN);sumup_packed_fp32_store(t##R##3,y3[i+R],VLEN); \
            sumup_packed_fp32_store(t##R##4,y4[i+R],VLEN);sumup_packed_fp32_store(t##R##5,y5[i+R],VLEN); \
            sumup_packed_fp32_store(t##R##6,y6[i+R],VLEN);sumup_packed_fp32_store(t##R##7,y7[i+R],VLEN); } while(0)
        ST(0); ST(1); ST(2); ST(3);
        #undef ST
    }
    for (; i < d; i++) {  /* row remainder: 1 row x 8 cols */
        __vr a0=_vel_vld_vssl(0,&zero[0],VLEN),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        bf16 *wp = w + (long)i*n;
        for (int j = 0; j < n; j += 2*VLEN) {
            const int vl = n - j < 2*VLEN ? (n - j) >> 1 : VLEN;
            __vr xv0,xv1,xv2,xv3,xv4,xv5,xv6,xv7,wv;
            LOADX_TRUNC(xv0,x0+j);LOADX_TRUNC(xv1,x1+j);LOADX_TRUNC(xv2,x2+j);LOADX_TRUNC(xv3,x3+j);
            LOADX_TRUNC(xv4,x4+j);LOADX_TRUNC(xv5,x5+j);LOADX_TRUNC(xv6,x6+j);LOADX_TRUNC(xv7,x7+j);
            UNPACK_W(wv,wp+j);
            a0=_vel_pvfmad_vvvvl(a0,xv0,wv,vl);a1=_vel_pvfmad_vvvvl(a1,xv1,wv,vl);
            a2=_vel_pvfmad_vvvvl(a2,xv2,wv,vl);a3=_vel_pvfmad_vvvvl(a3,xv3,wv,vl);
            a4=_vel_pvfmad_vvvvl(a4,xv4,wv,vl);a5=_vel_pvfmad_vvvvl(a5,xv5,wv,vl);
            a6=_vel_pvfmad_vvvvl(a6,xv6,wv,vl);a7=_vel_pvfmad_vvvvl(a7,xv7,wv,vl);
        }
        sumup_packed_fp32_store(a0,y0[i],VLEN);sumup_packed_fp32_store(a1,y1[i],VLEN);
        sumup_packed_fp32_store(a2,y2[i],VLEN);sumup_packed_fp32_store(a3,y3[i],VLEN);
        sumup_packed_fp32_store(a4,y4[i],VLEN);sumup_packed_fp32_store(a5,y5[i],VLEN);
        sumup_packed_fp32_store(a6,y6[i],VLEN);sumup_packed_fp32_store(a7,y7[i],VLEN);
    }
    #undef FMA8
    (void)xp; (void)yp;
}

void sgemv_packed_bf16_unr(float *y, float *x, bf16 *w, int n, int d) {
    int i;
    float zero[2] = {0.0f, 0.0f};
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffff, VLEN);
    /* Truncate F32 input to BF16 precision before the FMA. CPU's GGML
     * does this via vec_dot_type[BF16] = BF16 (it converts src1 from F32
     * to BF16 via from_float before the dot). Matching that exactly here
     * makes VE bit-perfect against the CPU baseline -- without this,
     * Qwen3.5's GDN block (which is L2_NORM-precision-sensitive)
     * diverges past ~token 20. Inline cost: one ANDM per j-chunk amortized
     * across 16 FMAs, so effectively free. */
    __vr bf16inputmsk = _vel_vbrdl_vsl(0xffff0000ffff0000UL, VLEN);

    if (d >= 16) {
        for (i = 0; i < d; i+=16) {
            __vr xv, xlv;
            __vr wv1, wv2, wv3, wv4, wv5, wv6, wv7, wv8;
            __vr wv9, wv10, wv11, wv12, wv13, wv14, wv15, wv16;
        
            __vr tv1 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv2 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv3 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv4 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv5 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv6 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv7 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv8 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv9 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv10 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv11 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv12 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv13 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv14 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv15 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv16 = _vel_vld_vssl(0, &zero[0], VLEN);

            bf16 *wp1 = w + i * n;
            bf16 *wp2 = w + (i + 1) * n;
            bf16 *wp3 = w + (i + 2) * n;
            bf16 *wp4 = w + (i + 3) * n;
            bf16 *wp5 = w + (i + 4) * n;
            bf16 *wp6 = w + (i + 5) * n;
            bf16 *wp7 = w + (i + 6) * n;
            bf16 *wp8 = w + (i + 7) * n;
            bf16 *wp9 = w + (i + 8) * n;
            bf16 *wp10 = w + (i + 9) * n;
            bf16 *wp11 = w + (i + 10) * n;
            bf16 *wp12 = w + (i + 11) * n;
            bf16 *wp13 = w + (i + 12) * n;
            bf16 *wp14 = w + (i + 13) * n;
            bf16 *wp15 = w + (i + 14) * n;
            bf16 *wp16 = w + (i + 15) * n;
            for (int j = 0; j < n; j += 2*VLEN) {
                const int vl = n - j < 2*VLEN ? (n - j)>>1 : VLEN;

                if ((unsigned long)(x+j) & 0x7) {
                    xv = _vel_vldu_vssl(8, (void *)(x + j + 1), vl);
                    xlv = _vel_vldlzx_vssl(8, (void *)(x + j), vl);
                    xv = _vel_pvor_vvvl(xv, xlv, vl);
                } else {
                    xv = _vel_vld_vssl(8, (void *)(x + j), vl);
                }
                xv = _vel_vand_vvvl(xv, bf16inputmsk, vl);

                load_bf16_to_packed_fp32(wv1,wp1+j,vl);
                load_bf16_to_packed_fp32(wv2,wp2+j,vl);
                load_bf16_to_packed_fp32(wv3,wp3+j,vl);
                load_bf16_to_packed_fp32(wv4,wp4+j,vl);
                load_bf16_to_packed_fp32(wv5,wp5+j,vl);
                load_bf16_to_packed_fp32(wv6,wp6+j,vl);
                load_bf16_to_packed_fp32(wv7,wp7+j,vl);
                load_bf16_to_packed_fp32(wv8,wp8+j,vl);
                load_bf16_to_packed_fp32(wv9,wp9+j,vl);
                load_bf16_to_packed_fp32(wv10,wp10+j,vl);
                load_bf16_to_packed_fp32(wv11,wp11+j,vl);
                load_bf16_to_packed_fp32(wv12,wp12+j,vl);
                load_bf16_to_packed_fp32(wv13,wp13+j,vl);
                load_bf16_to_packed_fp32(wv14,wp14+j,vl);
                load_bf16_to_packed_fp32(wv15,wp15+j,vl);
                load_bf16_to_packed_fp32(wv16,wp16+j,vl);

                tv1 = _vel_pvfmad_vvvvl(tv1, xv, wv1, vl);
                tv2 = _vel_pvfmad_vvvvl(tv2, xv, wv2, vl);
                tv3 = _vel_pvfmad_vvvvl(tv3, xv, wv3, vl);
                tv4 = _vel_pvfmad_vvvvl(tv4, xv, wv4, vl);
                tv5 = _vel_pvfmad_vvvvl(tv5, xv, wv5, vl);
                tv6 = _vel_pvfmad_vvvvl(tv6, xv, wv6, vl);
                tv7 = _vel_pvfmad_vvvvl(tv7, xv, wv7, vl);
                tv8 = _vel_pvfmad_vvvvl(tv8, xv, wv8, vl);
                tv9 = _vel_pvfmad_vvvvl(tv9, xv, wv9, vl);
                tv10 = _vel_pvfmad_vvvvl(tv10, xv, wv10, vl);
                tv11 = _vel_pvfmad_vvvvl(tv11, xv, wv11, vl);
                tv12 = _vel_pvfmad_vvvvl(tv12, xv, wv12, vl);
                tv13 = _vel_pvfmad_vvvvl(tv13, xv, wv13, vl);
                tv14 = _vel_pvfmad_vvvvl(tv14, xv, wv14, vl);
                tv15 = _vel_pvfmad_vvvvl(tv15, xv, wv15, vl);
                tv16 = _vel_pvfmad_vvvvl(tv16, xv, wv16, vl);
            }
            sumup_packed_fp32_store(tv1,y[i],VLEN);        
            sumup_packed_fp32_store(tv2,y[i + 1],VLEN);        
            sumup_packed_fp32_store(tv3,y[i + 2],VLEN);        
            sumup_packed_fp32_store(tv4,y[i + 3],VLEN);        
            sumup_packed_fp32_store(tv5,y[i + 4],VLEN);        
            sumup_packed_fp32_store(tv6,y[i + 5],VLEN);        
            sumup_packed_fp32_store(tv7,y[i + 6],VLEN);        
            sumup_packed_fp32_store(tv8,y[i + 7],VLEN);        
            sumup_packed_fp32_store(tv9,y[i + 8],VLEN);        
            sumup_packed_fp32_store(tv10,y[i + 9],VLEN);        
            sumup_packed_fp32_store(tv11,y[i + 10],VLEN);        
            sumup_packed_fp32_store(tv12,y[i + 11],VLEN);        
            sumup_packed_fp32_store(tv13,y[i + 12],VLEN);        
            sumup_packed_fp32_store(tv14,y[i + 13],VLEN);        
            sumup_packed_fp32_store(tv15,y[i + 14],VLEN);        
            sumup_packed_fp32_store(tv16,y[i + 15],VLEN);        
        }
    }
    for (i = (d/16) * 16; i < d; i++) {
        __vr xv, xlv;
        __vr wv1;
        __vr tv1 = _vel_vld_vssl(0, &zero[0], VLEN);
        bf16 *wp1 = w + i * n;
        for (int j = 0; j < n; j += 2*VLEN) {
            const int vl = n - j < 2*VLEN ? (n - j)>>1 : VLEN;

            if ((unsigned long)(x+j) & 0x7) {
                //printf("x+j is unaligned: %p\n", (void *)(x+j));
                xv = _vel_vldu_vssl(8, (void *)(x + j + 1), vl);
                xlv = _vel_vldlzx_vssl(8, (void *)(x + j), vl);
                xv = _vel_pvor_vvvl(xv, xlv, vl);
            } else {
                xv = _vel_vld_vssl(8, (void *)(x + j), vl);
            }
            xv = _vel_vand_vvvl(xv, bf16inputmsk, vl);

            load_bf16_to_packed_fp32(wv1,wp1+j,vl);

            tv1 = _vel_pvfmad_vvvvl(tv1, xv, wv1, vl);
        }
        sumup_packed_fp32_store(tv1,y[i],VLEN);        
    }
}

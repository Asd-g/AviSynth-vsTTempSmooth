#include "immintrin.h"
#include "vsTTempSmooth.h"

#if 0

#define unpck_ymm32_to_4ymm8(ymm_src32, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2) \
    ymm_l8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_0_3_4_7);         \
    ymm_l8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_8_11_12_15);      \
    ymm_h8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_16_19_20_23);     \
    ymm_h8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_24_27_28_31);     \
                                                                                \
    ymm_l8_1 = _mm256_unpacklo_epi8(ymm_l8_1, ymm_zero);                        \
    ymm_l8_2 = _mm256_unpacklo_epi8(ymm_l8_2, ymm_zero);                        \
    ymm_h8_1 = _mm256_unpacklo_epi8(ymm_h8_1, ymm_zero);                        \
    ymm_h8_2 = _mm256_unpacklo_epi8(ymm_h8_2, ymm_zero);                        \
                                                                                \
    ymm_l8_1 = _mm256_unpacklo_epi16(ymm_l8_1, ymm_zero);                       \
    ymm_l8_2 = _mm256_unpacklo_epi16(ymm_l8_2, ymm_zero);                       \
    ymm_h8_1 = _mm256_unpacklo_epi16(ymm_h8_1, ymm_zero);                       \
    ymm_h8_2 = _mm256_unpacklo_epi16(ymm_h8_2, ymm_zero);

#define pck_4ymm8_to_ymm32(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out32) \
    ymm_out_l8_1 = _mm256_packus_epi32(ymm_out_l8_1, ymm_zero);                               \
    ymm_out_l8_2 = _mm256_packus_epi32(ymm_out_l8_2, ymm_zero);                               \
    ymm_out_h8_1 = _mm256_packus_epi32(ymm_out_h8_1, ymm_zero);                               \
    ymm_out_h8_2 = _mm256_packus_epi32(ymm_out_h8_2, ymm_zero);                               \
                                                                                              \
    ymm_out_l8_1 = _mm256_packus_epi16(ymm_out_l8_1, ymm_zero);                               \
    ymm_out_l8_2 = _mm256_packus_epi16(ymm_out_l8_2, ymm_zero);                               \
    ymm_out_h8_1 = _mm256_packus_epi16(ymm_out_h8_1, ymm_zero);                               \
    ymm_out_h8_2 = _mm256_packus_epi16(ymm_out_h8_2, ymm_zero);                               \
                                                                                              \
    ymm_out_l8_1 = _mm256_permutevar8x32_epi32(ymm_out_l8_1, ymm_idx_4_7);                    \
    ymm_out_l8_2 = _mm256_permutevar8x32_epi32(ymm_out_l8_2, ymm_idx_12_15);                  \
    ymm_out_h8_1 = _mm256_permutevar8x32_epi32(ymm_out_h8_1, ymm_idx_20_23);                  \
    ymm_out_h8_2 = _mm256_permutevar8x32_epi32(ymm_out_h8_2, ymm_idx_28_31);                  \
                                                                                              \
    ymm_out32 = _mm256_blend_epi32(ymm_out_l8_1, ymm_out_l8_2, 0x0C);                         \
    ymm_out32 = _mm256_blend_epi32(ymm_out32, ymm_out_h8_1, 0x30);                            \
    ymm_out32 = _mm256_blend_epi32(ymm_out32, ymm_out_h8_2, 0xC0);

#define unpck_2ymm16_to_4ymm8(ymm_src16_1, ymm_src16_2, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2) \
    ymm_l8_1 = _mm256_permute4x64_epi64(ymm_src16_1, 0x10);                                     \
    ymm_l8_2 = _mm256_permute4x64_epi64(ymm_src16_1, 0x32);                                     \
                                                                                                \
    ymm_h8_1 = _mm256_permute4x64_epi64(ymm_src16_2, 0x10);                                     \
    ymm_h8_2 = _mm256_permute4x64_epi64(ymm_src16_2, 0x32);                                     \
                                                                                                \
    ymm_l8_1 = _mm256_unpacklo_epi16(ymm_l8_1, ymm_zero);                                       \
    ymm_l8_2 = _mm256_unpacklo_epi16(ymm_l8_2, ymm_zero);                                       \
                                                                                                \
    ymm_h8_1 = _mm256_unpacklo_epi16(ymm_h8_1, ymm_zero);                                       \
    ymm_h8_2 = _mm256_unpacklo_epi16(ymm_h8_2, ymm_zero);

#define pck_4ymm8_to_2ymm16(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out16_1, ymm_out16_2) \
    ymm_out16_1 = _mm256_packus_epi32(ymm_out_l8_1, ymm_out_l8_2);                                            \
    ymm_out16_2 = _mm256_packus_epi32(ymm_out_h8_1, ymm_out_h8_2);                                            \
                                                                                                              \
    ymm_out16_1 = _mm256_permute4x64_epi64(ymm_out16_1, 0xD8);                                                \
    ymm_out16_2 = _mm256_permute4x64_epi64(ymm_out16_2, 0xD8);


template<bool pfclip, bool fp>
template<typename T>
void TTempSmooth<pfclip, fp>::filterI_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{
    alignas(32) __m256i Temp256[(MAX_TEMP_RAD * 2 + 1) * 4];
    __m256i* pTemp256 = &Temp256[0];

    int src_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    int pf_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    const size_t stride{ dst->GetPitch(plane) / sizeof(T) };
    const size_t width{ dst->GetRowSize(plane) / sizeof(T) };
    const int height{ dst->GetHeight(plane) };
    const T* srcp[(MAX_TEMP_RAD * 2 + 1)]{}, * pfp[(MAX_TEMP_RAD * 2 + 1)]{};

    const int l{ plane >> 1 };
    const int thresh{ _thresh[l] << _shift };

    const int thUPD{ _thUPD[l] << _shift };
    const int pnew{ _pnew[l] << _shift };
    T* g_pMem{ reinterpret_cast<T*>(pIIRMem[l].data()) };
    int* g_pMemSum{ pMinSumMem[l].data() };
    const int iMaxSumDM{ (sizeof(T) < 2) ? 255 * (_maxr * 2 + 1) : 65535 * (_maxr * 2 + 1) };

    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / sizeof(T);
        pf_stride[i] = pf[i]->GetPitch(plane) / sizeof(T);
        srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
    }

    T* dstp{ reinterpret_cast<T*>(dst->GetWritePtr(plane)) };

    const __m256i ymm_zero = _mm256_setzero_si256();
    const __m256i ymm_idx_0_3_4_7 = _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 0);
    const __m256i ymm_idx_8_11_12_15 = _mm256_set_epi32(0, 0, 0, 3, 0, 0, 0, 2);
    const __m256i ymm_idx_16_19_20_23 = _mm256_set_epi32(0, 0, 0, 5, 0, 0, 0, 4);
    const __m256i ymm_idx_24_27_28_31 = _mm256_set_epi32(0, 0, 0, 7, 0, 0, 0, 6);

    const __m256i ymm_idx_4_7 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 4, 0);
    const __m256i ymm_idx_12_15 = _mm256_set_epi32(0, 0, 0, 0, 4, 0, 0, 0);
    const __m256i ymm_idx_20_23 = _mm256_set_epi32(0, 0, 4, 0, 0, 0, 0, 0);
    const __m256i ymm_idx_28_31 = _mm256_set_epi32(4, 0, 0, 0, 0, 0, 0, 0);

    const __m256i ymm_idx_mul = _mm256_set1_epi32(SIMD_AVX2_SPP);
    const __m256i ymm_idx_add_l8_1 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i ymm_idx_add_l8_2 = _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8);
    const __m256i ymm_idx_add_h8_1 = _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16);
    const __m256i ymm_idx_add_h8_2 = _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24);


    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 32)
        {
            // copy all input frames processed samples in SIMD pass in the temp buf in uint32 form
            for (int i = 0; i < (_maxr * 2 + 1); i++)
            {
                T* data_ptr;
                if (i == _maxr) // src sample
                {
                    data_ptr = (T*)&pfp[_maxr][x];
                }
                else // ref sample
                {
                    data_ptr = (T*)&srcp[i][x];
                }

                __m256i ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2;

                if (sizeof(T) == 1) // 8bit samples
                {
                    __m256i ymm_src32 = _mm256_load_si256((const __m256i*)data_ptr);

                    unpck_ymm32_to_4ymm8(ymm_src32, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2);

                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 0, ymm_l8_1);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 1, ymm_l8_2);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 2, ymm_h8_1);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 3, ymm_h8_2);
                }
                else // 16bit samples
                {
                    __m256i ymm_src16_1 = _mm256_load_si256((const __m256i*)data_ptr);
                    __m256i ymm_src16_2 = _mm256_load_si256((const __m256i*)(data_ptr + 16));

                    unpck_2ymm16_to_4ymm8(ymm_src16_1, ymm_src16_2, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2);

                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 0, ymm_l8_1);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 1, ymm_l8_2);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 2, ymm_h8_1);
                    _mm256_store_si256(pTemp256 + (int64_t)i * 4 + 3, ymm_h8_2);

                }

            }

            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m256i ymm_row_l8_1, ymm_row_l8_2, ymm_row_h8_1, ymm_row_h8_2;
            __m256i ymm_col_l8_1, ymm_col_l8_2, ymm_col_h8_1, ymm_col_h8_2;

            __m256i ymm_sum_minrow_l8_1 = _mm256_set1_epi32(iMaxSumDM);
            __m256i ymm_sum_minrow_l8_2 = _mm256_set1_epi32(iMaxSumDM);
            __m256i ymm_sum_minrow_h8_1 = _mm256_set1_epi32(iMaxSumDM);
            __m256i ymm_sum_minrow_h8_2 = _mm256_set1_epi32(iMaxSumDM);

            __m256i ymm_idx_minrow_l8_1 = _mm256_setzero_si256();
            __m256i ymm_idx_minrow_l8_2 = _mm256_setzero_si256();
            __m256i ymm_idx_minrow_h8_1 = _mm256_setzero_si256();
            __m256i ymm_idx_minrow_h8_2 = _mm256_setzero_si256();


            for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
            {
                __m256i ymm_sum_row_l8_1 = _mm256_setzero_si256();
                __m256i ymm_sum_row_l8_2 = _mm256_setzero_si256();
                __m256i ymm_sum_row_h8_1 = _mm256_setzero_si256();
                __m256i ymm_sum_row_h8_2 = _mm256_setzero_si256();

                for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                {
                    if (dmt_row == dmt_col)
                    { // samples with itselves => DM=0
                        continue;
                    }
                    __m256i* row_data_ptr = &pTemp256[dmt_row * 4];
                    __m256i* col_data_ptr = &pTemp256[dmt_col * 4];

                    ymm_row_l8_1 = _mm256_load_si256(row_data_ptr + 0);
                    ymm_row_l8_2 = _mm256_load_si256(row_data_ptr + 1);
                    ymm_row_h8_1 = _mm256_load_si256(row_data_ptr + 2);
                    ymm_row_h8_2 = _mm256_load_si256(row_data_ptr + 3);

                    ymm_col_l8_1 = _mm256_load_si256(col_data_ptr + 0);
                    ymm_col_l8_2 = _mm256_load_si256(col_data_ptr + 1);
                    ymm_col_h8_1 = _mm256_load_si256(col_data_ptr + 2);
                    ymm_col_h8_2 = _mm256_load_si256(col_data_ptr + 3);

                    __m256i ymm_subtr_l8_1 = _mm256_sub_epi32(ymm_row_l8_1, ymm_col_l8_1);
                    __m256i ymm_subtr_l8_2 = _mm256_sub_epi32(ymm_row_l8_2, ymm_col_l8_2);
                    __m256i ymm_subtr_h8_1 = _mm256_sub_epi32(ymm_row_h8_1, ymm_col_h8_1);
                    __m256i ymm_subtr_h8_2 = _mm256_sub_epi32(ymm_row_h8_2, ymm_col_h8_2);

                    __m256i ymm_abs_l8_1 = _mm256_abs_epi32(ymm_subtr_l8_1);
                    __m256i ymm_abs_l8_2 = _mm256_abs_epi32(ymm_subtr_l8_2);
                    __m256i ymm_abs_h8_1 = _mm256_abs_epi32(ymm_subtr_h8_1);
                    __m256i ymm_abs_h8_2 = _mm256_abs_epi32(ymm_subtr_h8_2);

                    ymm_sum_row_l8_1 = _mm256_add_epi32(ymm_sum_row_l8_1, ymm_abs_l8_1);
                    ymm_sum_row_l8_2 = _mm256_add_epi32(ymm_sum_row_l8_2, ymm_abs_l8_2);
                    ymm_sum_row_h8_1 = _mm256_add_epi32(ymm_sum_row_h8_1, ymm_abs_h8_1);
                    ymm_sum_row_h8_2 = _mm256_add_epi32(ymm_sum_row_h8_2, ymm_abs_h8_2);

                }

                __m256i ymm_mask_gt_l8_1 = _mm256_cmpgt_epi32(ymm_sum_minrow_l8_1, ymm_sum_row_l8_1);
                __m256i ymm_mask_gt_l8_2 = _mm256_cmpgt_epi32(ymm_sum_minrow_l8_2, ymm_sum_row_l8_2);
                __m256i ymm_mask_gt_h8_1 = _mm256_cmpgt_epi32(ymm_sum_minrow_h8_1, ymm_sum_row_h8_1);
                __m256i ymm_mask_gt_h8_2 = _mm256_cmpgt_epi32(ymm_sum_minrow_h8_2, ymm_sum_row_h8_2);

                __m256i ymm_idx_row = _mm256_set1_epi32(dmt_row);

                ymm_sum_minrow_l8_1 = _mm256_blendv_epi8(ymm_sum_minrow_l8_1, ymm_sum_row_l8_1, ymm_mask_gt_l8_1);
                ymm_sum_minrow_l8_2 = _mm256_blendv_epi8(ymm_sum_minrow_l8_2, ymm_sum_row_l8_2, ymm_mask_gt_l8_2);
                ymm_sum_minrow_h8_1 = _mm256_blendv_epi8(ymm_sum_minrow_h8_1, ymm_sum_row_h8_1, ymm_mask_gt_h8_1);
                ymm_sum_minrow_h8_2 = _mm256_blendv_epi8(ymm_sum_minrow_h8_2, ymm_sum_row_h8_2, ymm_mask_gt_h8_2);

                ymm_idx_minrow_l8_1 = _mm256_blendv_epi8(ymm_idx_minrow_l8_1, ymm_idx_row, ymm_mask_gt_l8_1);
                ymm_idx_minrow_l8_2 = _mm256_blendv_epi8(ymm_idx_minrow_l8_2, ymm_idx_row, ymm_mask_gt_l8_2);
                ymm_idx_minrow_h8_1 = _mm256_blendv_epi8(ymm_idx_minrow_h8_1, ymm_idx_row, ymm_mask_gt_h8_1);
                ymm_idx_minrow_h8_2 = _mm256_blendv_epi8(ymm_idx_minrow_h8_2, ymm_idx_row, ymm_mask_gt_h8_2);

            }

            ymm_idx_minrow_l8_1 = _mm256_mullo_epi32(ymm_idx_minrow_l8_1, ymm_idx_mul);
            ymm_idx_minrow_l8_2 = _mm256_mullo_epi32(ymm_idx_minrow_l8_2, ymm_idx_mul);
            ymm_idx_minrow_h8_1 = _mm256_mullo_epi32(ymm_idx_minrow_h8_1, ymm_idx_mul);
            ymm_idx_minrow_h8_2 = _mm256_mullo_epi32(ymm_idx_minrow_h8_2, ymm_idx_mul);

            ymm_idx_minrow_l8_1 = _mm256_add_epi32(ymm_idx_minrow_l8_1, ymm_idx_add_l8_1);
            ymm_idx_minrow_l8_2 = _mm256_add_epi32(ymm_idx_minrow_l8_2, ymm_idx_add_l8_2);
            ymm_idx_minrow_h8_1 = _mm256_add_epi32(ymm_idx_minrow_h8_1, ymm_idx_add_h8_1);
            ymm_idx_minrow_h8_2 = _mm256_add_epi32(ymm_idx_minrow_h8_2, ymm_idx_add_h8_2);

            __m256i ymm_best_l8_1 = _mm256_i32gather_epi32((int*)pTemp256, ymm_idx_minrow_l8_1, 4);
            __m256i ymm_best_l8_2 = _mm256_i32gather_epi32((int*)pTemp256, ymm_idx_minrow_l8_2, 4);
            __m256i ymm_best_h8_1 = _mm256_i32gather_epi32((int*)pTemp256, ymm_idx_minrow_h8_1, 4);
            __m256i ymm_best_h8_2 = _mm256_i32gather_epi32((int*)pTemp256, ymm_idx_minrow_h8_2, 4);

            // load and unpack pMem and pMemSum

            if (thUPD > 0) // IIR here)
            {
                __m256i ymm_Mem_l8_1, ymm_Mem_l8_2, ymm_Mem_h8_1, ymm_Mem_h8_2;

                if (sizeof(T) == 1) // 8bit samples
                {
                    __m256i ymm_Mem32 = _mm256_loadu_si256((const __m256i*) & pMem[x]);

                    unpck_ymm32_to_4ymm8(ymm_Mem32, ymm_Mem_l8_1, ymm_Mem_l8_2, ymm_Mem_h8_1, ymm_Mem_h8_2);

                }
                else // 16bit samples
                {
                    __m256i ymm_Mem16_1 = _mm256_loadu_si256((const __m256i*) & pMem[x]);
                    __m256i ymm_Mem16_2 = _mm256_loadu_si256((const __m256i*) & pMem[x + 16]); //?? 

                    unpck_2ymm16_to_4ymm8(ymm_Mem16_1, ymm_Mem16_2, ymm_Mem_l8_1, ymm_Mem_l8_2, ymm_Mem_h8_1, ymm_Mem_h8_2);
                }

                __m256i ymm_MemSum_l8_1, ymm_MemSum_l8_2, ymm_MemSum_h8_1, ymm_MemSum_h8_2;

                ymm_MemSum_l8_1 = _mm256_loadu_si256((const __m256i*) & pMemSum[x]); // todo: make pMem/pMemSum 32bytes aligned to use aligned load/store
                ymm_MemSum_l8_2 = _mm256_loadu_si256((const __m256i*) & pMemSum[x + 8]);
                ymm_MemSum_h8_1 = _mm256_loadu_si256((const __m256i*) & pMemSum[x + 16]);
                ymm_MemSum_h8_2 = _mm256_loadu_si256((const __m256i*) & pMemSum[x + 24]);

                // int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);
                __m256i ymm_dm_mem_l8_1 = _mm256_sub_epi32(ymm_best_l8_1, ymm_Mem_l8_1);
                __m256i ymm_dm_mem_l8_2 = _mm256_sub_epi32(ymm_best_l8_2, ymm_Mem_l8_2);
                __m256i ymm_dm_mem_h8_1 = _mm256_sub_epi32(ymm_best_h8_1, ymm_Mem_h8_1);
                __m256i ymm_dm_mem_h8_2 = _mm256_sub_epi32(ymm_best_h8_2, ymm_Mem_h8_2);

                ymm_dm_mem_l8_1 = _mm256_abs_epi32(ymm_dm_mem_l8_1);
                ymm_dm_mem_l8_2 = _mm256_abs_epi32(ymm_dm_mem_l8_2);
                ymm_dm_mem_h8_1 = _mm256_abs_epi32(ymm_dm_mem_h8_1);
                ymm_dm_mem_h8_2 = _mm256_abs_epi32(ymm_dm_mem_h8_2);

                //if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) > pMemSum[x + sub_x]))
                __m256i ymm_pnew = _mm256_set1_epi32(pnew);

                __m256i ymm_minsum_pnew_l8_1 = _mm256_add_epi32(ymm_sum_minrow_l8_1, ymm_pnew);
                __m256i ymm_minsum_pnew_l8_2 = _mm256_add_epi32(ymm_sum_minrow_l8_2, ymm_pnew);
                __m256i ymm_minsum_pnew_h8_1 = _mm256_add_epi32(ymm_sum_minrow_h8_1, ymm_pnew);
                __m256i ymm_minsum_pnew_h8_2 = _mm256_add_epi32(ymm_sum_minrow_h8_2, ymm_pnew);

                __m256i ymm_thUPD = _mm256_set1_epi32(thUPD);

                __m256i ymm_mask1_l8_1 = _mm256_cmpgt_epi32(ymm_thUPD, ymm_dm_mem_l8_1); // if (thUPD > dm_mem) = 1
                __m256i ymm_mask1_l8_2 = _mm256_cmpgt_epi32(ymm_thUPD, ymm_dm_mem_l8_2);
                __m256i ymm_mask1_h8_1 = _mm256_cmpgt_epi32(ymm_thUPD, ymm_dm_mem_h8_1);
                __m256i ymm_mask1_h8_2 = _mm256_cmpgt_epi32(ymm_thUPD, ymm_dm_mem_h8_2);

                __m256i ymm_mask2_l8_1 = _mm256_cmpgt_epi32(ymm_minsum_pnew_l8_1, ymm_MemSum_l8_1); // if (minsum_pnew > MemSum) = 1
                __m256i ymm_mask2_l8_2 = _mm256_cmpgt_epi32(ymm_minsum_pnew_l8_2, ymm_MemSum_l8_2);
                __m256i ymm_mask2_h8_1 = _mm256_cmpgt_epi32(ymm_minsum_pnew_h8_1, ymm_MemSum_h8_1);
                __m256i ymm_mask2_h8_2 = _mm256_cmpgt_epi32(ymm_minsum_pnew_h8_2, ymm_MemSum_h8_2);

                __m256i ymm_mask12_l8_1 = _mm256_and_si256(ymm_mask1_l8_1, ymm_mask2_l8_1);
                __m256i ymm_mask12_l8_2 = _mm256_and_si256(ymm_mask1_l8_2, ymm_mask2_l8_2);
                __m256i ymm_mask12_h8_1 = _mm256_and_si256(ymm_mask1_h8_1, ymm_mask2_h8_1);
                __m256i ymm_mask12_h8_2 = _mm256_and_si256(ymm_mask1_h8_2, ymm_mask2_h8_2);

                //mem still good - output mem block
                //best_data_ptr = &pMem[x + sub_x];
                ymm_best_l8_1 = _mm256_blendv_epi8(ymm_best_l8_1, ymm_Mem_l8_1, ymm_mask12_l8_1);
                ymm_best_l8_2 = _mm256_blendv_epi8(ymm_best_l8_2, ymm_Mem_l8_2, ymm_mask12_l8_2);
                ymm_best_h8_1 = _mm256_blendv_epi8(ymm_best_h8_1, ymm_Mem_h8_1, ymm_mask12_h8_1);
                ymm_best_h8_2 = _mm256_blendv_epi8(ymm_best_h8_2, ymm_Mem_h8_2, ymm_mask12_h8_2);

                // mem no good - update mem
                //pMem[x + sub_x] = *best_data_ptr;
                //pMemSum[x + sub_x] = i_sum_minrow;
                ymm_Mem_l8_1 = _mm256_blendv_epi8(ymm_best_l8_1, ymm_Mem_l8_1, ymm_mask12_l8_1);
                ymm_Mem_l8_2 = _mm256_blendv_epi8(ymm_best_l8_2, ymm_Mem_l8_2, ymm_mask12_l8_2);
                ymm_Mem_h8_1 = _mm256_blendv_epi8(ymm_best_h8_1, ymm_Mem_h8_1, ymm_mask12_h8_1);
                ymm_Mem_h8_2 = _mm256_blendv_epi8(ymm_best_h8_2, ymm_Mem_h8_2, ymm_mask12_h8_2);

                ymm_MemSum_l8_1 = _mm256_blendv_epi8(ymm_sum_minrow_l8_1, ymm_MemSum_l8_1, ymm_mask12_l8_1);
                ymm_MemSum_l8_2 = _mm256_blendv_epi8(ymm_sum_minrow_l8_2, ymm_MemSum_l8_2, ymm_mask12_l8_2);
                ymm_MemSum_h8_1 = _mm256_blendv_epi8(ymm_sum_minrow_h8_1, ymm_MemSum_h8_1, ymm_mask12_h8_1);
                ymm_MemSum_h8_2 = _mm256_blendv_epi8(ymm_sum_minrow_h8_2, ymm_MemSum_h8_2, ymm_mask12_h8_2);

                if (sizeof(T) == 1) // 8bit samples
                {
                    __m256i ymm_Mem_out32;
                    pck_4ymm8_to_ymm32(ymm_Mem_l8_1, ymm_Mem_l8_2, ymm_Mem_h8_1, ymm_Mem_h8_2, ymm_Mem_out32)

                        _mm256_storeu_si256((__m256i*)(&pMem[x]), ymm_Mem_out32);
                }
                else // 16bit samples
                {
                    __m256i ymm_Mem_out16_1, ymm_Mem_out16_2;

                    pck_4ymm8_to_2ymm16(ymm_Mem_l8_1, ymm_Mem_l8_2, ymm_Mem_h8_1, ymm_Mem_h8_2, ymm_Mem_out16_1, ymm_Mem_out16_2)

                        _mm256_storeu_si256((__m256i*)(&pMem[x]), ymm_Mem_out16_1);
                    _mm256_storeu_si256((__m256i*)(&pMem[x + 16]), ymm_Mem_out16_2);

                }

                _mm256_storeu_si256((__m256i*)(&pMemSum[x]), ymm_MemSum_l8_1);
                _mm256_storeu_si256((__m256i*)(&pMemSum[x + 8]), ymm_MemSum_l8_2);
                _mm256_storeu_si256((__m256i*)(&pMemSum[x + 16]), ymm_MemSum_h8_1);
                _mm256_storeu_si256((__m256i*)(&pMemSum[x + 24]), ymm_MemSum_h8_2);

            }

            // process in 32bit to reuse stored unpacked src ?

            __m256i* src_data_ptr = &pTemp256[_maxr * 4];

            __m256i ymm_src_l8_1 = _mm256_load_si256(src_data_ptr + 0);
            __m256i ymm_src_l8_2 = _mm256_load_si256(src_data_ptr + 1);
            __m256i ymm_src_h8_1 = _mm256_load_si256(src_data_ptr + 2);
            __m256i ymm_src_h8_2 = _mm256_load_si256(src_data_ptr + 3);

            __m256i ymm_subtr_l8_1 = _mm256_sub_epi32(ymm_best_l8_1, ymm_src_l8_1);
            __m256i ymm_subtr_l8_2 = _mm256_sub_epi32(ymm_best_l8_2, ymm_src_l8_2);
            __m256i ymm_subtr_h8_1 = _mm256_sub_epi32(ymm_best_h8_1, ymm_src_h8_1);
            __m256i ymm_subtr_h8_2 = _mm256_sub_epi32(ymm_best_h8_2, ymm_src_h8_2);

            __m256i ymm_abs_bs_l8_1 = _mm256_abs_epi32(ymm_subtr_l8_1);
            __m256i ymm_abs_bs_l8_2 = _mm256_abs_epi32(ymm_subtr_l8_2);
            __m256i ymm_abs_bs_h8_1 = _mm256_abs_epi32(ymm_subtr_h8_1);
            __m256i ymm_abs_bs_h8_2 = _mm256_abs_epi32(ymm_subtr_h8_2);

            __m256i ymm_thresh = _mm256_set1_epi32(thresh);

            __m256i ymm_mask_bs_gt_l8_1 = _mm256_cmpgt_epi32(ymm_abs_bs_l8_1, ymm_thresh);
            __m256i ymm_mask_bs_gt_l8_2 = _mm256_cmpgt_epi32(ymm_abs_bs_l8_2, ymm_thresh);
            __m256i ymm_mask_bs_gt_h8_1 = _mm256_cmpgt_epi32(ymm_abs_bs_h8_1, ymm_thresh);
            __m256i ymm_mask_bs_gt_h8_2 = _mm256_cmpgt_epi32(ymm_abs_bs_h8_2, ymm_thresh);

            __m256i ymm_out_l8_1 = _mm256_blendv_epi8(ymm_best_l8_1, ymm_src_l8_1, ymm_mask_bs_gt_l8_1);
            __m256i ymm_out_l8_2 = _mm256_blendv_epi8(ymm_best_l8_2, ymm_src_l8_2, ymm_mask_bs_gt_l8_2);
            __m256i ymm_out_h8_1 = _mm256_blendv_epi8(ymm_best_h8_1, ymm_src_h8_1, ymm_mask_bs_gt_h8_1);
            __m256i ymm_out_h8_2 = _mm256_blendv_epi8(ymm_best_h8_2, ymm_src_h8_2, ymm_mask_bs_gt_h8_2);

            if (sizeof(T) == 1) // 8bit samples
            {
                __m256i ymm_out32;
                pck_4ymm8_to_ymm32(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out32)

                    T* pDst = &dstp[x];
                _mm256_store_si256((__m256i*)(pDst), ymm_out32);
            }
            else // 16bit samples
            {
                __m256i ymm_out16_1, ymm_out16_2;

                pck_4ymm8_to_2ymm16(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out16_1, ymm_out16_2)

                    T* pDst = &dstp[x];
                _mm256_store_si256((__m256i*)(pDst), ymm_out16_1);
                _mm256_store_si256((__m256i*)(pDst + 16), ymm_out16_2); // ptr in shorts

            }
        }

        for (int i{ 0 }; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
        pMem += width;// mem_stride; ??
        pMemSum += width;
    }

}

template void TTempSmooth<true, true>::filterI_mode2_avx2<uint8_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterI_mode2_avx2<uint8_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterI_mode2_avx2<uint8_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterI_mode2_avx2<uint8_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);

template void TTempSmooth<true, true>::filterI_mode2_avx2<uint16_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterI_mode2_avx2<uint16_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterI_mode2_avx2<uint16_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterI_mode2_avx2<uint16_t>(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);

#endif

template<bool pfclip, bool fp>
void TTempSmooth<pfclip, fp>::filterF_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{
    int src_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    int pf_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    const int stride{dst->GetPitch(plane) / 4};
    const int width{dst->GetRowSize(plane) / 4};
    const int height{dst->GetHeight(plane)};
    const float *g_srcp[(MAX_TEMP_RAD * 2 + 1)]{}, *g_pfp[(MAX_TEMP_RAD * 2 + 1)]{};

    const int l{plane >> 1};
    const float thresh{(_thresh[l] / 256.0f)};

    const float thUPD{(_thUPD[l] / 256.0f)};
    const float pnew{(_pnew[l] / 256.0f)};
    float* g_pMem{reinterpret_cast<float*>(pIIRMem[l].data())};
    float* g_pMemSum{reinterpret_cast<float*>(pMinSumMem[l].data())};
    const float fMaxSumDM{2.0f};

    for (int i{0}; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / 4;
        pf_stride[i] = pf[i]->GetPitch(plane) / 4;
        g_srcp[i] = reinterpret_cast<const float*>(src[i]->GetReadPtr(plane));
        g_pfp[i] = reinterpret_cast<const float*>(pf[i]->GetReadPtr(plane));
    }

    float* g_dstp{reinterpret_cast<float*>(dst->GetWritePtr(plane))};

    const __m512 sign_bit = _mm512_set1_ps(-0.0f);
    const __m256 sign_bit_256 = _mm256_set1_ps(-0.0f);

    const __m256i ymm_idx_mul = _mm256_set1_epi32(SIMD_AVX2_SPP / 2); // 16 samples process here
    const __m256i ymm_idx_add_l8_1 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i ymm_idx_add_l8_2 = _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8);

    const __m512i zmm_idx_mul = _mm512_set1_epi32(SIMD_AVX512_SPP);
    const __m512i zmm_idx_add_l16_1 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i zmm_idx_add_l16_2 = _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);
    const __m512i zmm_idx_add_h16_1 = _mm512_set_epi32(47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32);
    const __m512i zmm_idx_add_h16_2 = _mm512_set_epi32(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48);

#pragma omp parallel for num_threads(_threads)
    for (int y = 0; y < height; ++y)
    {
        // local thread temps
        alignas(64) __m512 Temp512[(MAX_TEMP_RAD * 2 + 1) * 4];
        __m512* pTemp512 = &Temp512[0];

        alignas(32) __m256 Temp256[(MAX_TEMP_RAD * 2 + 1) * 2];
        __m256* pTemp256 = &Temp256[0];

        const int col64 = width - (width % SIMD_AVX512_SPP);

        // local threads ptrs
        const float *srcp[(MAX_TEMP_RAD * 2 + 1)]{}, *pfp[(MAX_TEMP_RAD * 2 + 1)]{};
        float *dstp, *pMem;
        float* pMemSum;

        for (int i{0}; i < _diameter; ++i)
        {
            srcp[i] = g_srcp[i] + y * src_stride[i];
            pfp[i] = g_pfp[i] + y * pf_stride[i];
        }

        dstp = g_dstp + y * stride;
        pMem = g_pMem + y * width;
        pMemSum = g_pMemSum + y * width;

        for (int x{0}; x < col64; x += SIMD_AVX512_SPP)
        {
            // copy all input frames processed samples in SIMD pass in the temp buf in float32 form
            for (int i = 0; i < (_maxr * 2 + 1); i++)
            {
                float* data_ptr;
                if (i == _maxr) // src sample
                {
                    data_ptr = (float*)&pfp[_maxr][x];
                }
                else // ref sample
                {
                    data_ptr = (float*)&srcp[i][x];
                }

                __m512 zmm_l16_1, zmm_l16_2, zmm_h16_1, zmm_h16_2;

                zmm_l16_1 = _mm512_load_ps(data_ptr);
                zmm_l16_2 = _mm512_load_ps((data_ptr + 16));
                zmm_h16_1 = _mm512_load_ps((data_ptr + 32));
                zmm_h16_2 = _mm512_load_ps((data_ptr + 48));

                _mm512_store_ps((float*)(pTemp512 + (int64_t)i * 4 + 0), zmm_l16_1);
                _mm512_store_ps((float*)(pTemp512 + (int64_t)i * 4 + 1), zmm_l16_2);
                _mm512_store_ps((float*)(pTemp512 + (int64_t)i * 4 + 2), zmm_h16_1);
                _mm512_store_ps((float*)(pTemp512 + (int64_t)i * 4 + 3), zmm_h16_2);
            }

            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m512 zmm_row_l16_1, zmm_row_l16_2, zmm_row_h16_1, zmm_row_h16_2;
            __m512 zmm_col_l16_1, zmm_col_l16_2, zmm_col_h16_1, zmm_col_h16_2;

            __m512 zmm_sum_minrow_l16_1 = _mm512_set1_ps(fMaxSumDM);
            __m512 zmm_sum_minrow_l16_2 = _mm512_set1_ps(fMaxSumDM);
            __m512 zmm_sum_minrow_h16_1 = _mm512_set1_ps(fMaxSumDM);
            __m512 zmm_sum_minrow_h16_2 = _mm512_set1_ps(fMaxSumDM);

            __m512i zmm_idx_minrow_l16_1 = _mm512_setzero_si512();
            __m512i zmm_idx_minrow_l16_2 = _mm512_setzero_si512();
            __m512i zmm_idx_minrow_h16_1 = _mm512_setzero_si512();
            __m512i zmm_idx_minrow_h16_2 = _mm512_setzero_si512();

            for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
            {
                __m512 zmm_sum_row_l16_1 = _mm512_setzero_ps();
                __m512 zmm_sum_row_l16_2 = _mm512_setzero_ps();
                __m512 zmm_sum_row_h16_1 = _mm512_setzero_ps();
                __m512 zmm_sum_row_h16_2 = _mm512_setzero_ps();

                for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                {
                    if (dmt_row == dmt_col)
                    { // samples with itselves => DM=0
                        continue;
                    }
                    __m512* row_data_ptr = &pTemp512[dmt_row * 4];
                    __m512* col_data_ptr = &pTemp512[dmt_col * 4];

                    zmm_row_l16_1 = _mm512_load_ps((float*)(row_data_ptr + 0));
                    zmm_row_l16_2 = _mm512_load_ps((float*)(row_data_ptr + 1));
                    zmm_row_h16_1 = _mm512_load_ps((float*)(row_data_ptr + 2));
                    zmm_row_h16_2 = _mm512_load_ps((float*)(row_data_ptr + 3));

                    zmm_col_l16_1 = _mm512_load_ps((float*)(col_data_ptr + 0));
                    zmm_col_l16_2 = _mm512_load_ps((float*)(col_data_ptr + 1));
                    zmm_col_h16_1 = _mm512_load_ps((float*)(col_data_ptr + 2));
                    zmm_col_h16_2 = _mm512_load_ps((float*)(col_data_ptr + 3));

                    __m512 zmm_subtr_l16_1 = _mm512_sub_ps(zmm_row_l16_1, zmm_col_l16_1);
                    __m512 zmm_subtr_l16_2 = _mm512_sub_ps(zmm_row_l16_2, zmm_col_l16_2);
                    __m512 zmm_subtr_h16_1 = _mm512_sub_ps(zmm_row_h16_1, zmm_col_h16_1);
                    __m512 zmm_subtr_h16_2 = _mm512_sub_ps(zmm_row_h16_2, zmm_col_h16_2);

                    __m512 zmm_abs_l16_1 = _mm512_andnot_ps(sign_bit, zmm_subtr_l16_1); // or _mm512_abs_ps() Latency 8 ??
                    __m512 zmm_abs_l16_2 = _mm512_andnot_ps(sign_bit, zmm_subtr_l16_2);
                    __m512 zmm_abs_h16_1 = _mm512_andnot_ps(sign_bit, zmm_subtr_h16_1);
                    __m512 zmm_abs_h16_2 = _mm512_andnot_ps(sign_bit, zmm_subtr_h16_2);

                    zmm_sum_row_l16_1 = _mm512_add_ps(zmm_sum_row_l16_1, zmm_abs_l16_1);
                    zmm_sum_row_l16_2 = _mm512_add_ps(zmm_sum_row_l16_2, zmm_abs_l16_2);
                    zmm_sum_row_h16_1 = _mm512_add_ps(zmm_sum_row_h16_1, zmm_abs_h16_1);
                    zmm_sum_row_h16_2 = _mm512_add_ps(zmm_sum_row_h16_2, zmm_abs_h16_2);
                }

                __mmask16 kmm_mask_gt_l16_1 = _mm512_cmp_ps_mask(zmm_sum_minrow_l16_1, zmm_sum_row_l16_1, _CMP_GT_OQ);
                __mmask16 kmm_mask_gt_l16_2 = _mm512_cmp_ps_mask(zmm_sum_minrow_l16_2, zmm_sum_row_l16_2, _CMP_GT_OQ);
                __mmask16 kmm_mask_gt_h16_1 = _mm512_cmp_ps_mask(zmm_sum_minrow_h16_1, zmm_sum_row_h16_1, _CMP_GT_OQ);
                __mmask16 kmm_mask_gt_h16_2 = _mm512_cmp_ps_mask(zmm_sum_minrow_h16_2, zmm_sum_row_h16_2, _CMP_GT_OQ);

                __m512i zmm_idx_row = _mm512_set1_epi32(dmt_row);

                zmm_sum_minrow_l16_1 = _mm512_mask_blend_ps(kmm_mask_gt_l16_1, zmm_sum_minrow_l16_1, zmm_sum_row_l16_1);
                zmm_sum_minrow_l16_2 = _mm512_mask_blend_ps(kmm_mask_gt_l16_2, zmm_sum_minrow_l16_2, zmm_sum_row_l16_2);
                zmm_sum_minrow_h16_1 = _mm512_mask_blend_ps(kmm_mask_gt_h16_1, zmm_sum_minrow_h16_1, zmm_sum_row_h16_1);
                zmm_sum_minrow_h16_2 = _mm512_mask_blend_ps(kmm_mask_gt_h16_2, zmm_sum_minrow_h16_2, zmm_sum_row_h16_2);

                zmm_idx_minrow_l16_1 = _mm512_mask_blend_epi32(kmm_mask_gt_l16_1, zmm_idx_minrow_l16_1, zmm_idx_row);
                zmm_idx_minrow_l16_2 = _mm512_mask_blend_epi32(kmm_mask_gt_l16_2, zmm_idx_minrow_l16_2, zmm_idx_row);
                zmm_idx_minrow_h16_1 = _mm512_mask_blend_epi32(kmm_mask_gt_h16_1, zmm_idx_minrow_h16_1, zmm_idx_row);
                zmm_idx_minrow_h16_2 = _mm512_mask_blend_epi32(kmm_mask_gt_h16_2, zmm_idx_minrow_h16_2, zmm_idx_row);
            }

            zmm_idx_minrow_l16_1 = _mm512_mullo_epi32(zmm_idx_minrow_l16_1, zmm_idx_mul);
            zmm_idx_minrow_l16_2 = _mm512_mullo_epi32(zmm_idx_minrow_l16_2, zmm_idx_mul);
            zmm_idx_minrow_h16_1 = _mm512_mullo_epi32(zmm_idx_minrow_h16_1, zmm_idx_mul);
            zmm_idx_minrow_h16_2 = _mm512_mullo_epi32(zmm_idx_minrow_h16_2, zmm_idx_mul);

            zmm_idx_minrow_l16_1 = _mm512_add_epi32(zmm_idx_minrow_l16_1, zmm_idx_add_l16_1);
            zmm_idx_minrow_l16_2 = _mm512_add_epi32(zmm_idx_minrow_l16_2, zmm_idx_add_l16_2);
            zmm_idx_minrow_h16_1 = _mm512_add_epi32(zmm_idx_minrow_h16_1, zmm_idx_add_h16_1);
            zmm_idx_minrow_h16_2 = _mm512_add_epi32(zmm_idx_minrow_h16_2, zmm_idx_add_h16_2);

            __m512 zmm_best_l16_1 = _mm512_i32gather_ps(zmm_idx_minrow_l16_1, (float*)pTemp512, 4);
            __m512 zmm_best_l16_2 = _mm512_i32gather_ps(zmm_idx_minrow_l16_2, (float*)pTemp512, 4);
            __m512 zmm_best_h16_1 = _mm512_i32gather_ps(zmm_idx_minrow_h16_1, (float*)pTemp512, 4);
            __m512 zmm_best_h16_2 = _mm512_i32gather_ps(zmm_idx_minrow_h16_2, (float*)pTemp512, 4);

            // load and unpack pMem and pMemSum

            if (thUPD > 0) // IIR here)
            {
                __m512 zmm_Mem_l16_1, zmm_Mem_l16_2, zmm_Mem_h16_1, zmm_Mem_h16_2;

                zmm_Mem_l16_1 = _mm512_loadu_ps(&pMem[x]);
                zmm_Mem_l16_2 = _mm512_loadu_ps(&pMem[x + 16]);
                zmm_Mem_h16_1 = _mm512_loadu_ps(&pMem[x + 32]);
                zmm_Mem_h16_2 = _mm512_loadu_ps(&pMem[x + 48]);

                __m512 zmm_MemSum_l16_1, zmm_MemSum_l16_2, zmm_MemSum_h16_1, zmm_MemSum_h16_2;

                zmm_MemSum_l16_1 = _mm512_loadu_ps(&pMemSum[x]); // todo: make pMem/pMemSum 32bytes aligned to use aligned load/store
                zmm_MemSum_l16_2 = _mm512_loadu_ps(&pMemSum[x + 16]);
                zmm_MemSum_h16_1 = _mm512_loadu_ps(&pMemSum[x + 32]);
                zmm_MemSum_h16_2 = _mm512_loadu_ps(&pMemSum[x + 48]);

                // int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);
                __m512 zmm_dm_mem_l16_1 = _mm512_sub_ps(zmm_best_l16_1, zmm_Mem_l16_1);
                __m512 zmm_dm_mem_l16_2 = _mm512_sub_ps(zmm_best_l16_2, zmm_Mem_l16_2);
                __m512 zmm_dm_mem_h16_1 = _mm512_sub_ps(zmm_best_h16_1, zmm_Mem_h16_1);
                __m512 zmm_dm_mem_h16_2 = _mm512_sub_ps(zmm_best_h16_2, zmm_Mem_h16_2);

                zmm_dm_mem_l16_1 = _mm512_andnot_ps(sign_bit, zmm_dm_mem_l16_1);
                zmm_dm_mem_l16_2 = _mm512_andnot_ps(sign_bit, zmm_dm_mem_l16_2);
                zmm_dm_mem_h16_1 = _mm512_andnot_ps(sign_bit, zmm_dm_mem_h16_1);
                zmm_dm_mem_h16_2 = _mm512_andnot_ps(sign_bit, zmm_dm_mem_h16_2);

                // if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) > pMemSum[x + sub_x]))
                __m512 zmm_pnew = _mm512_set1_ps(pnew);

                __m512 zmm_minsum_pnew_l16_1 = _mm512_add_ps(zmm_sum_minrow_l16_1, zmm_pnew);
                __m512 zmm_minsum_pnew_l16_2 = _mm512_add_ps(zmm_sum_minrow_l16_2, zmm_pnew);
                __m512 zmm_minsum_pnew_h16_1 = _mm512_add_ps(zmm_sum_minrow_h16_1, zmm_pnew);
                __m512 zmm_minsum_pnew_h16_2 = _mm512_add_ps(zmm_sum_minrow_h16_2, zmm_pnew);

                __m512 zmm_thUPD = _mm512_set1_ps(thUPD);

                __mmask16 kmm_mask1_l16_1 = _mm512_cmp_ps_mask(zmm_thUPD, zmm_dm_mem_l16_1, _CMP_GT_OQ); // if (thUPD > dm_mem) = 1
                __mmask16 kmm_mask1_l16_2 = _mm512_cmp_ps_mask(zmm_thUPD, zmm_dm_mem_l16_2, _CMP_GT_OQ);
                __mmask16 kmm_mask1_h16_1 = _mm512_cmp_ps_mask(zmm_thUPD, zmm_dm_mem_h16_1, _CMP_GT_OQ);
                __mmask16 kmm_mask1_h16_2 = _mm512_cmp_ps_mask(zmm_thUPD, zmm_dm_mem_h16_2, _CMP_GT_OQ);

                /* __mmask16 ymm_mask2_l16_1 = _mm512_cmp_ps_mask(zmm_minsum_pnew_l16_1, zmm_MemSum_l16_1, _CMP_GT_OQ); // if (minsum_pnew >
                MemSum) = 1
                __mmask16 ymm_mask2_l16_2 = _mm512_cmp_ps_mask(zmm_minsum_pnew_l16_2, zmm_MemSum_l16_2, _CMP_GT_OQ);
                __mmask16 ymm_mask2_h16_1 = _mm512_cmp_ps_mask(zmm_minsum_pnew_h16_1, zmm_MemSum_h16_1, _CMP_GT_OQ);
                __mmask16 ymm_mask2_h16_2 = _mm512_cmp_ps_mask(zmm_minsum_pnew_h16_2, zmm_MemSum_h16_2, _CMP_GT_OQ);
                */

                __mmask16 kmm_mask2_l16_1 = _mm512_mask_cmp_ps_mask(
                    kmm_mask1_l16_1, zmm_minsum_pnew_l16_1, zmm_MemSum_l16_1, _CMP_GT_OQ); // if (minsum_pnew > MemSum) = 1
                __mmask16 kmm_mask2_l16_2 = _mm512_mask_cmp_ps_mask(kmm_mask1_l16_2, zmm_minsum_pnew_l16_2, zmm_MemSum_l16_2, _CMP_GT_OQ);
                __mmask16 kmm_mask2_h16_1 = _mm512_mask_cmp_ps_mask(kmm_mask1_h16_1, zmm_minsum_pnew_h16_1, zmm_MemSum_h16_1, _CMP_GT_OQ);
                __mmask16 kmm_mask2_h16_2 = _mm512_mask_cmp_ps_mask(kmm_mask1_h16_2, zmm_minsum_pnew_h16_2, zmm_MemSum_h16_2, _CMP_GT_OQ);

                /* __m256 ymm_mask12_l8_1 = _mm256_and_ps(ymm_mask1_l8_1, ymm_mask2_l8_1);
                __m256 ymm_mask12_l8_2 = _mm256_and_ps(ymm_mask1_l8_2, ymm_mask2_l8_2);
                __m256 ymm_mask12_h8_1 = _mm256_and_ps(ymm_mask1_h8_1, ymm_mask2_h8_1);
                __m256 ymm_mask12_h8_2 = _mm256_and_ps(ymm_mask1_h8_2, ymm_mask2_h8_2);
                */

                // mem still good - output mem block
                // best_data_ptr = &pMem[x + sub_x];
                zmm_best_l16_1 = _mm512_mask_blend_ps(kmm_mask2_l16_1, zmm_best_l16_1, zmm_Mem_l16_1);
                zmm_best_l16_2 = _mm512_mask_blend_ps(kmm_mask2_l16_2, zmm_best_l16_2, zmm_Mem_l16_2);
                zmm_best_h16_1 = _mm512_mask_blend_ps(kmm_mask2_h16_1, zmm_best_h16_1, zmm_Mem_h16_1);
                zmm_best_h16_2 = _mm512_mask_blend_ps(kmm_mask2_h16_2, zmm_best_h16_2, zmm_Mem_h16_2);

                // mem no good - update mem
                // pMem[x + sub_x] = *best_data_ptr;
                // pMemSum[x + sub_x] = i_sum_minrow;
                zmm_Mem_l16_1 = _mm512_mask_blend_ps(kmm_mask2_l16_1, zmm_best_l16_1, zmm_Mem_l16_1);
                zmm_Mem_l16_2 = _mm512_mask_blend_ps(kmm_mask2_l16_2, zmm_best_l16_2, zmm_Mem_l16_2);
                zmm_Mem_h16_1 = _mm512_mask_blend_ps(kmm_mask2_h16_1, zmm_best_h16_1, zmm_Mem_h16_1);
                zmm_Mem_h16_2 = _mm512_mask_blend_ps(kmm_mask2_h16_2, zmm_best_h16_2, zmm_Mem_h16_2);

                zmm_MemSum_l16_1 = _mm512_mask_blend_ps(kmm_mask2_l16_1, zmm_sum_minrow_l16_1, zmm_MemSum_l16_1);
                zmm_MemSum_l16_2 = _mm512_mask_blend_ps(kmm_mask2_l16_2, zmm_sum_minrow_l16_2, zmm_MemSum_l16_2);
                zmm_MemSum_h16_1 = _mm512_mask_blend_ps(kmm_mask2_h16_1, zmm_sum_minrow_h16_1, zmm_MemSum_h16_1);
                zmm_MemSum_h16_2 = _mm512_mask_blend_ps(kmm_mask2_h16_2, zmm_sum_minrow_h16_2, zmm_MemSum_h16_2);

                _mm512_storeu_ps((&pMem[x]), zmm_Mem_l16_1);
                _mm512_storeu_ps((&pMem[x + 16]), zmm_Mem_l16_2);
                _mm512_storeu_ps((&pMem[x + 32]), zmm_Mem_h16_1);
                _mm512_storeu_ps((&pMem[x + 48]), zmm_Mem_h16_2);

                _mm512_storeu_ps((&pMemSum[x]), zmm_MemSum_l16_1);
                _mm512_storeu_ps((&pMemSum[x + 16]), zmm_MemSum_l16_2);
                _mm512_storeu_ps((&pMemSum[x + 32]), zmm_MemSum_h16_1);
                _mm512_storeu_ps((&pMemSum[x + 48]), zmm_MemSum_h16_2);

                /* _mm512_mask_storeu_ps((&pMem[x]), kmm_mask2_l16_1, zmm_best_l16_1); - mask need invert, todo: make inverted mask in 2
                compare to use masked storage _mm512_mask_storeu_ps((&pMem[x + 16]), kmm_mask2_l16_2, zmm_best_l16_2);
                _mm512_mask_storeu_ps((&pMem[x + 32]), kmm_mask2_h16_1, zmm_best_h16_1);
                _mm512_mask_storeu_ps((&pMem[x + 48]), kmm_mask2_h16_2, zmm_best_h16_2);

                _mm512_mask_storeu_ps((&pMemSum[x]), kmm_mask2_l16_1, zmm_sum_minrow_l16_1);
                _mm512_mask_storeu_ps((&pMemSum[x + 16]), kmm_mask2_l16_2, zmm_sum_minrow_l16_2);
                _mm512_mask_storeu_ps((&pMemSum[x + 32]), kmm_mask2_h16_1, zmm_sum_minrow_h16_1);
                _mm512_mask_storeu_ps((&pMemSum[x + 48]), kmm_mask2_h16_2, zmm_sum_minrow_h16_2);
                */
            }

            // process in 32bit to reuse stored unpacked src ?

            __m512* src_data_ptr = &pTemp512[_maxr * 4];

            __m512 zmm_src_l16_1 = _mm512_load_ps((float*)(src_data_ptr + 0));
            __m512 zmm_src_l16_2 = _mm512_load_ps((float*)(src_data_ptr + 1));
            __m512 zmm_src_h16_1 = _mm512_load_ps((float*)(src_data_ptr + 2));
            __m512 zmm_src_h16_2 = _mm512_load_ps((float*)(src_data_ptr + 3));

            __m512 zmm_subtr_l16_1 = _mm512_sub_ps(zmm_best_l16_1, zmm_src_l16_1);
            __m512 zmm_subtr_l16_2 = _mm512_sub_ps(zmm_best_l16_2, zmm_src_l16_2);
            __m512 zmm_subtr_h16_1 = _mm512_sub_ps(zmm_best_h16_1, zmm_src_h16_1);
            __m512 zmm_subtr_h16_2 = _mm512_sub_ps(zmm_best_h16_2, zmm_src_h16_2);

            __m512 zmm_abs_bs_l16_1 = _mm512_andnot_ps(sign_bit, zmm_subtr_l16_1);
            __m512 zmm_abs_bs_l16_2 = _mm512_andnot_ps(sign_bit, zmm_subtr_l16_2);
            __m512 zmm_abs_bs_h16_1 = _mm512_andnot_ps(sign_bit, zmm_subtr_h16_1);
            __m512 zmm_abs_bs_h16_2 = _mm512_andnot_ps(sign_bit, zmm_subtr_h16_2);

            __m512 zmm_thresh = _mm512_set1_ps(thresh);

            __mmask16 kmm_mask_bs_gt_l16_1 = _mm512_cmp_ps_mask(zmm_abs_bs_l16_1, zmm_thresh, _CMP_GT_OQ);
            __mmask16 kmm_mask_bs_gt_l16_2 = _mm512_cmp_ps_mask(zmm_abs_bs_l16_2, zmm_thresh, _CMP_GT_OQ);
            __mmask16 kmm_mask_bs_gt_h16_1 = _mm512_cmp_ps_mask(zmm_abs_bs_h16_1, zmm_thresh, _CMP_GT_OQ);
            __mmask16 kmm_mask_bs_gt_h16_2 = _mm512_cmp_ps_mask(zmm_abs_bs_h16_2, zmm_thresh, _CMP_GT_OQ);

            __m512 zmm_out_l16_1 = _mm512_mask_blend_ps(kmm_mask_bs_gt_l16_1, zmm_best_l16_1, zmm_src_l16_1);
            __m512 zmm_out_l16_2 = _mm512_mask_blend_ps(kmm_mask_bs_gt_l16_2, zmm_best_l16_2, zmm_src_l16_2);
            __m512 zmm_out_h16_1 = _mm512_mask_blend_ps(kmm_mask_bs_gt_h16_1, zmm_best_h16_1, zmm_src_h16_1);
            __m512 zmm_out_h16_2 = _mm512_mask_blend_ps(kmm_mask_bs_gt_h16_2, zmm_best_h16_2, zmm_src_h16_2);

            float* pDst = &dstp[x];
            _mm512_store_ps((pDst), zmm_out_l16_1);
            _mm512_store_ps((pDst + 16), zmm_out_l16_2);
            _mm512_store_ps((pDst + 32), zmm_out_h16_1);
            _mm512_store_ps((pDst + 48), zmm_out_h16_2);
        }

        // process last columns with 16 AVX2 - may be make it inline function to reuse from _AVX2.cpp ?
        for (int x{col64}; x < width; x += 16)
        {
            // copy all input frames processed samples in SIMD pass in the temp buf in float32 form
            for (int i = 0; i < (_maxr * 2 + 1); i++)
            {
                float* data_ptr;
                if (i == _maxr) // src sample
                {
                    data_ptr = (float*)&pfp[_maxr][x];
                }
                else // ref sample
                {
                    data_ptr = (float*)&srcp[i][x];
                }

                __m256 ymm_l8_1, ymm_l8_2;

                ymm_l8_1 = _mm256_load_ps(data_ptr);
                ymm_l8_2 = _mm256_load_ps((data_ptr + 8));

                _mm256_store_ps((float*)(pTemp256 + (int64_t)i * 2 + 0), ymm_l8_1);
                _mm256_store_ps((float*)(pTemp256 + (int64_t)i * 2 + 1), ymm_l8_2);
            }

            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m256 ymm_row_l8_1, ymm_row_l8_2;
            __m256 ymm_col_l8_1, ymm_col_l8_2;

            __m256 ymm_sum_minrow_l8_1 = _mm256_set1_ps(fMaxSumDM);
            __m256 ymm_sum_minrow_l8_2 = _mm256_set1_ps(fMaxSumDM);

            __m256i ymm_idx_minrow_l8_1 = _mm256_setzero_si256();
            __m256i ymm_idx_minrow_l8_2 = _mm256_setzero_si256();

            for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
            {
                __m256 ymm_sum_row_l8_1 = _mm256_setzero_ps();
                __m256 ymm_sum_row_l8_2 = _mm256_setzero_ps();

                for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                {
                    if (dmt_row == dmt_col)
                    { // samples with itselves => DM=0
                        continue;
                    }
                    __m256* row_data_ptr = &pTemp256[dmt_row * 2];
                    __m256* col_data_ptr = &pTemp256[dmt_col * 2];

                    ymm_row_l8_1 = _mm256_load_ps((float*)(row_data_ptr + 0));
                    ymm_row_l8_2 = _mm256_load_ps((float*)(row_data_ptr + 1));

                    ymm_col_l8_1 = _mm256_load_ps((float*)(col_data_ptr + 0));
                    ymm_col_l8_2 = _mm256_load_ps((float*)(col_data_ptr + 1));

                    __m256 ymm_subtr_l8_1 = _mm256_sub_ps(ymm_row_l8_1, ymm_col_l8_1);
                    __m256 ymm_subtr_l8_2 = _mm256_sub_ps(ymm_row_l8_2, ymm_col_l8_2);

                    __m256 ymm_abs_l8_1 = _mm256_andnot_ps(sign_bit_256, ymm_subtr_l8_1);
                    __m256 ymm_abs_l8_2 = _mm256_andnot_ps(sign_bit_256, ymm_subtr_l8_2);

                    ymm_sum_row_l8_1 = _mm256_add_ps(ymm_sum_row_l8_1, ymm_abs_l8_1);
                    ymm_sum_row_l8_2 = _mm256_add_ps(ymm_sum_row_l8_2, ymm_abs_l8_2);
                }

                __m256 ymm_mask_gt_l8_1 = _mm256_cmp_ps(ymm_sum_minrow_l8_1, ymm_sum_row_l8_1, _CMP_GT_OQ);
                __m256 ymm_mask_gt_l8_2 = _mm256_cmp_ps(ymm_sum_minrow_l8_2, ymm_sum_row_l8_2, _CMP_GT_OQ);

                __m256i ymm_idx_row = _mm256_set1_epi32(dmt_row);

                ymm_sum_minrow_l8_1 = _mm256_blendv_ps(ymm_sum_minrow_l8_1, ymm_sum_row_l8_1, ymm_mask_gt_l8_1);
                ymm_sum_minrow_l8_2 = _mm256_blendv_ps(ymm_sum_minrow_l8_2, ymm_sum_row_l8_2, ymm_mask_gt_l8_2);

                ymm_idx_minrow_l8_1 = _mm256_blendv_epi8(ymm_idx_minrow_l8_1, ymm_idx_row, _mm256_castps_si256(ymm_mask_gt_l8_1));
                ymm_idx_minrow_l8_2 = _mm256_blendv_epi8(ymm_idx_minrow_l8_2, ymm_idx_row, _mm256_castps_si256(ymm_mask_gt_l8_2));
            }

            ymm_idx_minrow_l8_1 = _mm256_mullo_epi32(ymm_idx_minrow_l8_1, ymm_idx_mul);
            ymm_idx_minrow_l8_2 = _mm256_mullo_epi32(ymm_idx_minrow_l8_2, ymm_idx_mul);

            ymm_idx_minrow_l8_1 = _mm256_add_epi32(ymm_idx_minrow_l8_1, ymm_idx_add_l8_1);
            ymm_idx_minrow_l8_2 = _mm256_add_epi32(ymm_idx_minrow_l8_2, ymm_idx_add_l8_2);

            __m256 ymm_best_l8_1 = _mm256_i32gather_ps((float*)pTemp256, ymm_idx_minrow_l8_1, 4);
            __m256 ymm_best_l8_2 = _mm256_i32gather_ps((float*)pTemp256, ymm_idx_minrow_l8_2, 4);

            // load and unpack pMem and pMemSum

            if (thUPD > 0) // IIR here)
            {
                __m256 ymm_Mem_l8_1, ymm_Mem_l8_2;

                ymm_Mem_l8_1 = _mm256_loadu_ps(&pMem[x]);
                ymm_Mem_l8_2 = _mm256_loadu_ps(&pMem[x + 8]);

                __m256 ymm_MemSum_l8_1, ymm_MemSum_l8_2;

                ymm_MemSum_l8_1 = _mm256_loadu_ps(&pMemSum[x]); // todo: make pMem/pMemSum 32bytes aligned to use aligned load/store
                ymm_MemSum_l8_2 = _mm256_loadu_ps(&pMemSum[x + 8]);

                // int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);
                __m256 ymm_dm_mem_l8_1 = _mm256_sub_ps(ymm_best_l8_1, ymm_Mem_l8_1);
                __m256 ymm_dm_mem_l8_2 = _mm256_sub_ps(ymm_best_l8_2, ymm_Mem_l8_2);

                ymm_dm_mem_l8_1 = _mm256_andnot_ps(sign_bit_256, ymm_dm_mem_l8_1);
                ymm_dm_mem_l8_2 = _mm256_andnot_ps(sign_bit_256, ymm_dm_mem_l8_2);

                // if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) > pMemSum[x + sub_x]))
                __m256 ymm_pnew = _mm256_set1_ps(pnew);

                __m256 ymm_minsum_pnew_l8_1 = _mm256_add_ps(ymm_sum_minrow_l8_1, ymm_pnew);
                __m256 ymm_minsum_pnew_l8_2 = _mm256_add_ps(ymm_sum_minrow_l8_2, ymm_pnew);

                __m256 ymm_thUPD = _mm256_set1_ps(thUPD);

                __m256 ymm_mask1_l8_1 = _mm256_cmp_ps(ymm_thUPD, ymm_dm_mem_l8_1, _CMP_GT_OQ); // if (thUPD > dm_mem) = 1
                __m256 ymm_mask1_l8_2 = _mm256_cmp_ps(ymm_thUPD, ymm_dm_mem_l8_2, _CMP_GT_OQ);

                __m256 ymm_mask2_l8_1 = _mm256_cmp_ps(ymm_minsum_pnew_l8_1, ymm_MemSum_l8_1, _CMP_GT_OQ); // if (minsum_pnew > MemSum) = 1
                __m256 ymm_mask2_l8_2 = _mm256_cmp_ps(ymm_minsum_pnew_l8_2, ymm_MemSum_l8_2, _CMP_GT_OQ);

                __m256 ymm_mask12_l8_1 = _mm256_and_ps(ymm_mask1_l8_1, ymm_mask2_l8_1);
                __m256 ymm_mask12_l8_2 = _mm256_and_ps(ymm_mask1_l8_2, ymm_mask2_l8_2);

                // mem still good - output mem block
                // best_data_ptr = &pMem[x + sub_x];
                ymm_best_l8_1 = _mm256_blendv_ps(ymm_best_l8_1, ymm_Mem_l8_1, ymm_mask12_l8_1);
                ymm_best_l8_2 = _mm256_blendv_ps(ymm_best_l8_2, ymm_Mem_l8_2, ymm_mask12_l8_2);

                // mem no good - update mem
                // pMem[x + sub_x] = *best_data_ptr;
                // pMemSum[x + sub_x] = i_sum_minrow;
                ymm_Mem_l8_1 = _mm256_blendv_ps(ymm_best_l8_1, ymm_Mem_l8_1, ymm_mask12_l8_1);
                ymm_Mem_l8_2 = _mm256_blendv_ps(ymm_best_l8_2, ymm_Mem_l8_2, ymm_mask12_l8_2);

                ymm_MemSum_l8_1 = _mm256_blendv_ps(ymm_sum_minrow_l8_1, ymm_MemSum_l8_1, ymm_mask12_l8_1);
                ymm_MemSum_l8_2 = _mm256_blendv_ps(ymm_sum_minrow_l8_2, ymm_MemSum_l8_2, ymm_mask12_l8_2);

                _mm256_storeu_ps((&pMem[x]), ymm_Mem_l8_1);
                _mm256_storeu_ps((&pMem[x + 8]), ymm_Mem_l8_2);

                _mm256_storeu_ps((&pMemSum[x]), ymm_MemSum_l8_1);
                _mm256_storeu_ps((&pMemSum[x + 8]), ymm_MemSum_l8_2);
            }

            // process in 32bit to reuse stored unpacked src ?

            __m256* src_data_ptr = &pTemp256[_maxr * 2];

            __m256 ymm_src_l8_1 = _mm256_load_ps((float*)(src_data_ptr + 0));
            __m256 ymm_src_l8_2 = _mm256_load_ps((float*)(src_data_ptr + 1));

            __m256 ymm_subtr_l8_1 = _mm256_sub_ps(ymm_best_l8_1, ymm_src_l8_1);
            __m256 ymm_subtr_l8_2 = _mm256_sub_ps(ymm_best_l8_2, ymm_src_l8_2);

            __m256 ymm_abs_bs_l8_1 = _mm256_andnot_ps(sign_bit_256, ymm_subtr_l8_1);
            __m256 ymm_abs_bs_l8_2 = _mm256_andnot_ps(sign_bit_256, ymm_subtr_l8_2);

            __m256 ymm_thresh = _mm256_set1_ps(thresh);

            __m256 ymm_mask_bs_gt_l8_1 = _mm256_cmp_ps(ymm_abs_bs_l8_1, ymm_thresh, _CMP_GT_OQ);
            __m256 ymm_mask_bs_gt_l8_2 = _mm256_cmp_ps(ymm_abs_bs_l8_2, ymm_thresh, _CMP_GT_OQ);

            __m256 ymm_out_l8_1 = _mm256_blendv_ps(ymm_best_l8_1, ymm_src_l8_1, ymm_mask_bs_gt_l8_1);
            __m256 ymm_out_l8_2 = _mm256_blendv_ps(ymm_best_l8_2, ymm_src_l8_2, ymm_mask_bs_gt_l8_2);

            float* pDst = &dstp[x];
            _mm256_store_ps((pDst), ymm_out_l8_1);
            _mm256_store_ps((pDst + 8), ymm_out_l8_2);
        }
    }
}

template void TTempSmooth<true, true>::filterF_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterF_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterF_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterF_mode2_avx512(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);

#include "VCL2/vectorclass.h"
#include "vsTTempSmooth.h"


#define unpck_ymm32_to_4ymm8(ymm_src32, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2) \
ymm_l8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_0_3_4_7);\
ymm_l8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_8_11_12_15);\
ymm_h8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_16_19_20_23);\
ymm_h8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_24_27_28_31);\
\
ymm_l8_1 = _mm256_unpacklo_epi8(ymm_l8_1, ymm_zero);\
ymm_l8_2 = _mm256_unpacklo_epi8(ymm_l8_2, ymm_zero);\
ymm_h8_1 = _mm256_unpacklo_epi8(ymm_h8_1, ymm_zero);\
ymm_h8_2 = _mm256_unpacklo_epi8(ymm_h8_2, ymm_zero);\
\
ymm_l8_1 = _mm256_unpacklo_epi16(ymm_l8_1, ymm_zero);\
ymm_l8_2 = _mm256_unpacklo_epi16(ymm_l8_2, ymm_zero);\
ymm_h8_1 = _mm256_unpacklo_epi16(ymm_h8_1, ymm_zero);\
ymm_h8_2 = _mm256_unpacklo_epi16(ymm_h8_2, ymm_zero);

#define pck_4ymm8_to_ymm32(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out32)\
ymm_out_l8_1 = _mm256_packus_epi32(ymm_out_l8_1, ymm_zero);\
ymm_out_l8_2 = _mm256_packus_epi32(ymm_out_l8_2, ymm_zero);\
ymm_out_h8_1 = _mm256_packus_epi32(ymm_out_h8_1, ymm_zero);\
ymm_out_h8_2 = _mm256_packus_epi32(ymm_out_h8_2, ymm_zero);\
\
ymm_out_l8_1 = _mm256_packus_epi16(ymm_out_l8_1, ymm_zero);\
ymm_out_l8_2 = _mm256_packus_epi16(ymm_out_l8_2, ymm_zero);\
ymm_out_h8_1 = _mm256_packus_epi16(ymm_out_h8_1, ymm_zero);\
ymm_out_h8_2 = _mm256_packus_epi16(ymm_out_h8_2, ymm_zero);\
\
ymm_out_l8_1 = _mm256_permutevar8x32_epi32(ymm_out_l8_1, ymm_idx_4_7);\
ymm_out_l8_2 = _mm256_permutevar8x32_epi32(ymm_out_l8_2, ymm_idx_12_15);\
ymm_out_h8_1 = _mm256_permutevar8x32_epi32(ymm_out_h8_1, ymm_idx_20_23);\
ymm_out_h8_2 = _mm256_permutevar8x32_epi32(ymm_out_h8_2, ymm_idx_28_31);\
\
ymm_out32 = _mm256_blend_epi32(ymm_out_l8_1, ymm_out_l8_2, 0x0C);\
ymm_out32 = _mm256_blend_epi32(ymm_out32, ymm_out_h8_1, 0x30);\
ymm_out32 = _mm256_blend_epi32(ymm_out32, ymm_out_h8_2, 0xC0);

#define unpck_2ymm16_to_4ymm8(ymm_src16_1, ymm_src16_2, ymm_l8_1, ymm_l8_2, ymm_h8_1, ymm_h8_2)\
ymm_l8_1 = _mm256_permute4x64_epi64(ymm_src16_1, 0x10);\
ymm_l8_2 = _mm256_permute4x64_epi64(ymm_src16_1, 0x32);\
\
ymm_h8_1 = _mm256_permute4x64_epi64(ymm_src16_2, 0x10);\
ymm_h8_2 = _mm256_permute4x64_epi64(ymm_src16_2, 0x32);\
\
ymm_l8_1 = _mm256_unpacklo_epi16(ymm_l8_1, ymm_zero);\
ymm_l8_2 = _mm256_unpacklo_epi16(ymm_l8_2, ymm_zero);\
\
ymm_h8_1 = _mm256_unpacklo_epi16(ymm_h8_1, ymm_zero);\
ymm_h8_2 = _mm256_unpacklo_epi16(ymm_h8_2, ymm_zero);

#define pck_4ymm8_to_2ymm16(ymm_out_l8_1, ymm_out_l8_2, ymm_out_h8_1, ymm_out_h8_2, ymm_out16_1, ymm_out16_2)\
ymm_out16_1 = _mm256_packus_epi32(ymm_out_l8_1, ymm_out_l8_2);\
ymm_out16_2 = _mm256_packus_epi32(ymm_out_h8_1, ymm_out_h8_2);\
\
ymm_out16_1 = _mm256_permute4x64_epi64(ymm_out16_1, 0xD8);\
ymm_out16_2 = _mm256_permute4x64_epi64(ymm_out16_2, 0xD8);






template <typename T>
AVS_FORCEINLINE static Vec8i load(const void* p)
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return Vec8i().load_8uc(p);
    else
        return Vec8i().load_8us(p);
}

template <bool pfclip, bool fp>
template <typename T, bool useDiff>
void TTempSmooth<pfclip, fp>::filterI_avx2(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept
{
    int src_stride[15]{};
    int pf_stride[15]{};
    const size_t stride{ dst->GetPitch(plane) / sizeof(T) };
    const size_t width{ dst->GetRowSize(plane) / sizeof(T) };
    const int height{ dst->GetHeight(plane) };
    const T* srcp[15]{}, * pfp[15]{};
    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / sizeof(T);
        pf_stride[i] = pf[i]->GetPitch(plane) / sizeof(T);
        srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
    }

    T* __restrict dstp{ reinterpret_cast<T*>(dst->GetWritePtr(plane)) };

    const int l{ plane >> 1 };
    const float* const weightSaved{ _weight[l].data() };
    const Vec8i thresh{ _thresh[l] << _shift };

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        for (int y{ 0 }; y < height; ++y)
        {
            for (int x{ 0 }; x < width; x += 32)
            {
                const auto& c01{ load<T>(&pfp[_maxr][x]) };
                const auto& srcp_v01{ load<T>(&srcp[_maxr][x]) };

                const auto& c02{ load<T>(&pfp[_maxr][x + 8]) };
                const auto& srcp_v02{ load<T>(&srcp[_maxr][x + 8]) };

                const auto& c03{ load<T>(&pfp[_maxr][x + 16]) };
                const auto& srcp_v03{ load<T>(&srcp[_maxr][x + 16]) };

                const auto& c04{ load<T>(&pfp[_maxr][x + 24]) };
                const auto& srcp_v04{ load<T>(&srcp[_maxr][x + 24]) };

                Vec8f weights01{ _cw };
                auto sum01{ to_float(srcp_v01) * weights01 };

                Vec8f weights02{ _cw };
                auto sum02{ to_float(srcp_v02) * weights02 };

                Vec8f weights03{ _cw };
                auto sum03{ to_float(srcp_v03) * weights03 };

                Vec8f weights04{ _cw };
                auto sum04{ to_float(srcp_v04) * weights04 };

                int frameIndex{ _maxr - 1 };

                if (frameIndex > fromFrame)
                {
                    auto t1_01{ load<T>(&pfp[frameIndex][x]) };
                    auto diff01{ abs(c01 - t1_01) };
                    const auto check_v01{ diff01 < thresh };

                    auto t1_02{ load<T>(&pfp[frameIndex][x + 8]) };
                    auto diff02{ abs(c02 - t1_02) };
                    const auto check_v02{ diff02 < thresh };

                    auto t1_03{ load<T>(&pfp[frameIndex][x + 16]) };
                    auto diff03{ abs(c03 - t1_03) };
                    const auto check_v03{ diff03 < thresh };

                    auto t1_04{ load<T>(&pfp[frameIndex][x + 24]) };
                    auto diff04{ abs(c04 - t1_04) };
                    const auto check_v04{ diff04 < thresh };

                    auto weight01{ (useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight02{ (useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight03{ (useDiff) ? lookup<1792>(diff03 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight04{ (useDiff) ? lookup<1792>(diff04 >> _shift, weightSaved) : weightSaved[frameIndex] };

                    weights01 = select(Vec8fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec8fb(check_v02), weights02 + weight02, weights02);
                    weights03 = select(Vec8fb(check_v03), weights03 + weight03, weights03);
                    weights04 = select(Vec8fb(check_v04), weights04 + weight04, weights04);

                    sum01 = select(Vec8fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec8fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);
                    sum03 = select(Vec8fb(check_v03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight03, sum03), sum03);
                    sum04 = select(Vec8fb(check_v04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 24])), weight04, sum04), sum04);

                    --frameIndex;
                    int v{ 256 };

                    while (frameIndex > fromFrame)
                    {
                        const auto& t2_01{ t1_01 };
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{ diff01 < thresh && abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh && abs(t1_02 - t2_02) < thresh };

                        const auto& t2_03{ t1_03 };
                        t1_03 = load<T>(&pfp[frameIndex][x + 16]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{ diff03 < thresh && abs(t1_03 - t2_03) < thresh };

                        const auto& t2_04{ t1_04 };
                        t1_04 = load<T>(&pfp[frameIndex][x + 24]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{ diff04 < thresh && abs(t1_04 - t2_04) < thresh };

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight03 = (useDiff) ? lookup<1792>((diff03 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight04 = (useDiff) ? lookup<1792>((diff04 >> _shift) + v, weightSaved) : weightSaved[frameIndex];

                        weights01 = select(Vec8fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec8fb(check_v1_02), weights02 + weight02, weights02);
                        weights03 = select(Vec8fb(check_v1_03), weights03 + weight03, weights03);
                        weights04 = select(Vec8fb(check_v1_04), weights04 + weight04, weights04);

                        sum01 = select(Vec8fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec8fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);
                        sum03 = select(Vec8fb(check_v1_03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight03, sum03), sum03);
                        sum04 = select(Vec8fb(check_v1_04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 24])), weight04, sum04), sum04);

                        --frameIndex;
                        v += 256;
                    }
                }

                frameIndex = _maxr + 1;

                if (frameIndex < toFrame)
                {
                    auto t1_01{ load<T>(&pfp[frameIndex][x]) };
                    auto diff01{ abs(c01 - t1_01) };
                    const auto check_v01{ diff01 < thresh };

                    auto t1_02{ load<T>(&pfp[frameIndex][x + 8]) };
                    auto diff02{ abs(c02 - t1_02) };
                    const auto check_v02{ diff02 < thresh };

                    auto t1_03{ load<T>(&pfp[frameIndex][x + 16]) };
                    auto diff03{ abs(c03 - t1_03) };
                    const auto check_v03{ diff03 < thresh };

                    auto t1_04{ load<T>(&pfp[frameIndex][x + 24]) };
                    auto diff04{ abs(c04 - t1_04) };
                    const auto check_v04{ diff04 < thresh };

                    auto weight01{ (useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight02{ (useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight03{ (useDiff) ? lookup<1792>(diff03 >> _shift, weightSaved) : weightSaved[frameIndex] };
                    auto weight04{ (useDiff) ? lookup<1792>(diff04 >> _shift, weightSaved) : weightSaved[frameIndex] };

                    weights01 = select(Vec8fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec8fb(check_v02), weights02 + weight02, weights02);
                    weights03 = select(Vec8fb(check_v03), weights03 + weight03, weights03);
                    weights04 = select(Vec8fb(check_v04), weights04 + weight04, weights04);

                    sum01 = select(Vec8fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec8fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);
                    sum03 = select(Vec8fb(check_v03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight03, sum03), sum03);
                    sum04 = select(Vec8fb(check_v04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 24])), weight04, sum04), sum04);

                    ++frameIndex;
                    int v{ 256 };

                    while (frameIndex < toFrame)
                    {
                        const auto& t2_01{ t1_01 };
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{ diff01 < thresh && abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh && abs(t1_02 - t2_02) < thresh };

                        const auto& t2_03{ t1_03 };
                        t1_03 = load<T>(&pfp[frameIndex][x + 16]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{ diff03 < thresh && abs(t1_03 - t2_03) < thresh };

                        const auto& t2_04{ t1_04 };
                        t1_04 = load<T>(&pfp[frameIndex][x + 24]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{ diff04 < thresh && abs(t1_04 - t2_04) < thresh };

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight03 = (useDiff) ? lookup<1792>((diff03 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight04 = (useDiff) ? lookup<1792>((diff04 >> _shift) + v, weightSaved) : weightSaved[frameIndex];

                        weights01 = select(Vec8fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec8fb(check_v1_02), weights02 + weight02, weights02);
                        weights03 = select(Vec8fb(check_v1_03), weights03 + weight03, weights03);
                        weights04 = select(Vec8fb(check_v1_04), weights04 + weight04, weights04);

                        sum01 = select(Vec8fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec8fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);
                        sum03 = select(Vec8fb(check_v1_03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight03, sum03), sum03);
                        sum04 = select(Vec8fb(check_v1_04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 24])), weight04, sum04), sum04);

                        ++frameIndex;
                        v += 256;
                    }
                }

                if constexpr (fp)
                {
                    compress_saturated_s2u(compress_saturated(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x])), (1.0f - weights01), sum01 + 0.5f)), zero_si256()), zero_si256()).get_low().storel(dstp + x);
                    compress_saturated_s2u(compress_saturated(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 8])), (1.0f - weights02), sum02 + 0.5f)), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 8));
                    compress_saturated_s2u(compress_saturated(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 16])), (1.0f - weights03), sum03 + 0.5f)), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 16));
                    compress_saturated_s2u(compress_saturated(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 24])), (1.0f - weights04), sum04 + 0.5f)), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 24));
                }
                else
                {
                    compress_saturated_s2u(compress_saturated(truncatei(sum01 / weights01 + 0.5f), zero_si256()), zero_si256()).get_low().storel(dstp + x);
                    compress_saturated_s2u(compress_saturated(truncatei(sum02 / weights02 + 0.5f), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 8));
                    compress_saturated_s2u(compress_saturated(truncatei(sum03 / weights03 + 0.5f), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 16));
                    compress_saturated_s2u(compress_saturated(truncatei(sum04 / weights04 + 0.5f), zero_si256()), zero_si256()).get_low().storel(dstp + (x + 24));
                }
            }

            for (int i{ 0 }; i < _diameter; ++i)
            {
                srcp[i] += src_stride[i];
                pfp[i] += pf_stride[i];
            }

            dstp += stride;
        }
    }
    else
    {
        for (int y{ 0 }; y < height; ++y)
        {
            for (int x{ 0 }; x < width; x += 16)
            {
                const auto& c01{ load<T>(&pfp[_maxr][x]) };
                const auto& srcp_v01{ load<T>(&srcp[_maxr][x]) };

                const auto& c02{ load<T>(&pfp[_maxr][x + 8]) };
                const auto& srcp_v02{ load<T>(&srcp[_maxr][x + 8]) };

                Vec8f weights01{ _cw };
                auto sum01{ to_float(srcp_v01) * weights01 };

                Vec8f weights02{ _cw };
                auto sum02{ to_float(srcp_v02) * weights02 };

                int frameIndex{ _maxr - 1 };

                if (frameIndex > fromFrame)
                {
                    auto t1_01{ load<T>(&pfp[frameIndex][x]) };
                    auto diff01{ abs(c01 - t1_01) };
                    const auto check_v01{ diff01 < thresh };

                    auto t1_02{ load<T>(&pfp[frameIndex][x + 8]) };
                    auto diff02{ abs(c02 - t1_02) };
                    const auto check_v02{ diff02 < thresh };

                    auto weight01{ (useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };
                    auto weight02{ (useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };

                    weights01 = select(Vec8fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec8fb(check_v02), weights02 + weight02, weights02);

                    sum01 = select(Vec8fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec8fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);

                    --frameIndex;
                    int v{ 256 };

                    while (frameIndex > fromFrame)
                    {
                        const auto& t2_01{ t1_01 };
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{ diff01 < thresh && abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh && abs(t1_02 - t2_02) < thresh };

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);

                        weights01 = select(Vec8fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec8fb(check_v1_02), weights02 + weight02, weights02);

                        sum01 = select(Vec8fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec8fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);

                        --frameIndex;
                        v += 256;
                    }
                }

                frameIndex = _maxr + 1;

                if (frameIndex < toFrame)
                {
                    auto t1_01{ load<T>(&pfp[frameIndex][x]) };
                    auto diff01{ abs(c01 - t1_01) };
                    const auto check_v01{ diff01 < thresh };

                    auto t1_02{ load<T>(&pfp[frameIndex][x + 8]) };
                    auto diff02{ abs(c02 - t1_02) };
                    const auto check_v02{ diff02 < thresh };

                    auto weight01{ (useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };
                    auto weight02{ (useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };

                    weights01 = select(Vec8fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec8fb(check_v02), weights02 + weight02, weights02);

                    sum01 = select(Vec8fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec8fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);

                    ++frameIndex;
                    int v{ 256 };

                    while (frameIndex < toFrame)
                    {
                        const auto& t2_01{ t1_01 };
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{ diff01 < thresh && abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh && abs(t1_02 - t2_02) < thresh };

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);

                        weights01 = select(Vec8fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec8fb(check_v1_02), weights02 + weight02, weights02);

                        sum01 = select(Vec8fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec8fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 8])), weight02, sum02), sum02);

                        ++frameIndex;
                        v += 256;
                    }
                }

                if constexpr (fp)
                {
                    compress_saturated_s2u(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x])), (1.0f - weights01), sum01 + 0.5f)), zero_si256()).get_low().store(dstp + x);
                    compress_saturated_s2u(truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 8])), (1.0f - weights02), sum02 + 0.5f)), zero_si256()).get_low().store(dstp + (x + 8));
                }
                else
                {
                    compress_saturated_s2u(truncatei(sum01 / weights01 + 0.5f), zero_si256()).get_low().store(dstp + x);
                    compress_saturated_s2u(truncatei(sum02 / weights02 + 0.5f), zero_si256()).get_low().store(dstp + (x + 8));
                }
            }

            for (int i{ 0 }; i < _diameter; ++i)
            {
                srcp[i] += src_stride[i];
                pfp[i] += pf_stride[i];
            }

            dstp += stride;
        }
    }
}

template void TTempSmooth<true, true>::filterI_avx2<uint8_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx2<uint8_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterI_avx2<uint8_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx2<uint8_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterI_avx2<uint8_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx2<uint8_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterI_avx2<uint8_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx2<uint8_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<true, true>::filterI_avx2<uint16_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx2<uint16_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterI_avx2<uint16_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx2<uint16_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterI_avx2<uint16_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx2<uint16_t, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterI_avx2<uint16_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx2<uint16_t, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;


template<bool pfclip, bool fp>
template<typename T>
void TTempSmooth<pfclip, fp>::filterI_mode2_avx2(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{

#define SIMD_AVX2_SPP 32

    __m256i alignas(32) Temp256[(MAX_TEMP_RAD * 2 + 1) * 4];
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
    T* pMem = 0;
    if ((plane >> 1) == 0) pMem = reinterpret_cast<T*>(pIIRMemY);
    if ((plane >> 1) == 1) pMem = reinterpret_cast<T*>(pIIRMemU);
    if ((plane >> 1) == 2) pMem = reinterpret_cast<T*>(pIIRMemV);

    int* pMemSum = 0;
    if ((plane >> 1) == 0) pMemSum = pMinSumMemY;
    if ((plane >> 1) == 1) pMemSum = pMinSumMemU;
    if ((plane >> 1) == 2) pMemSum = pMinSumMemV;

    const int iMaxSumDM = (sizeof(T) < 2) ? 255 * (_maxr * 2 + 1) : 65535 * (_maxr * 2 + 1);

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



template <bool pfclip, bool fp>
template <bool useDiff>
void TTempSmooth<pfclip, fp>::filterF_avx2(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept
{
    int src_stride[15]{};
    int pf_stride[15]{};
    const int stride{ dst->GetPitch(plane) / 4 };
    const int width{ dst->GetRowSize(plane) / 4 };
    const int height{ dst->GetHeight(plane) };
    const float* srcp[15]{}, * pfp[15]{};
    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / 4;
        pf_stride[i] = pf[i]->GetPitch(plane) / 4;
        srcp[i] = reinterpret_cast<const float*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const float*>(pf[i]->GetReadPtr(plane));
    }

    float* __restrict dstp{ reinterpret_cast<float*>(dst->GetWritePtr(plane)) };

    const int l{ plane >> 1 };
    const float* const weightSaved{ _weight[l].data() };
    const Vec8f thresh{ _threshF[l] };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 8)
        {
            const auto& c{ Vec8f().load(&pfp[_maxr][x]) };
            const auto& srcp_v{ Vec8f().load(&srcp[_maxr][x]) };

            Vec8f weights{ _cw };
            auto sum{ srcp_v * weights };

            int frameIndex{ _maxr - 1 };

            if (frameIndex > fromFrame)
            {
                auto t1{ Vec8f().load(&pfp[frameIndex][x]) };
                auto diff{ min(abs(c - t1), 1.0f) };
                const auto check_v{ diff < thresh };

                auto weight{ (useDiff) ? lookup<1792>(truncatei(diff * 255.0f), weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };
                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, mul_add(Vec8f().load(&srcp[frameIndex][x]), weight, sum), sum);

                --frameIndex;
                int v{ 256 };

                while (frameIndex > fromFrame)
                {
                    const auto& t2{ t1 };
                    t1 = Vec8f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{ diff < thresh && min(abs(t1 - t2), 1.0f) < thresh };

                    weight = (useDiff) ? lookup<1792>(truncatei(diff * 255.0f) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);
                    weights = select(check_v1, weights + weight, weights);
                    sum = select(check_v1, mul_add(Vec8f().load(&srcp[frameIndex][x]), weight, sum), sum);

                    --frameIndex;
                    v += 256;
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                auto t1{ Vec8f().load(&pfp[frameIndex][x]) };
                auto diff{ min(abs(c - t1), 1.0f) };
                const auto check_v{ diff < thresh };

                auto weight{ (useDiff) ? lookup<1792>(truncatei(diff * 255.0f), weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved) };
                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, mul_add(Vec8f().load(&srcp[frameIndex][x]), weight, sum), sum);

                ++frameIndex;
                int v{ 256 };

                while (frameIndex < toFrame)
                {
                    const auto& t2{ t1 };
                    t1 = Vec8f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{ diff < thresh && min(abs(t1 - t2), 1.0f) < thresh };

                    weight = (useDiff) ? lookup<1792>(truncatei(diff * 255.0f) + v, weightSaved) : lookup<1792>(Vec8i(frameIndex), weightSaved);
                    weights = select(check_v1, weights + weight, weights);
                    sum = select(check_v1, mul_add(Vec8f().load(&srcp[frameIndex][x]), weight, sum), sum);

                    ++frameIndex;
                    v += 256;
                }
            }

            if constexpr (fp)
                mul_add(Vec8f().load(&srcp[_maxr][x]), (1.0f - weights), sum).store(dstp + x);
            else
                (sum / weights).store(dstp + x);
        }

        for (int i{ 0 }; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
    }
}

template void TTempSmooth<true, true>::filterF_avx2<true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterF_avx2<true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterF_avx2<false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterF_avx2<false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterF_avx2<true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterF_avx2<true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterF_avx2<false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterF_avx2<false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template <typename T>
float ComparePlane_avx2(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept
{
    const size_t pitch{ src->GetPitch(PLANAR_Y) / sizeof(T) };
    const size_t pitch2{ src1->GetPitch(PLANAR_Y) / sizeof(T) };
    const size_t width{ src->GetRowSize(PLANAR_Y) / sizeof(T) };
    const int height{ src->GetHeight(PLANAR_Y) };
    const T* srcp{ reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y)) };
    const T* srcp2{ reinterpret_cast<const T*>(src1->GetReadPtr(PLANAR_Y)) };

    Vec8f accum{ 0.0f };

    for (size_t y{ 0 }; y < height; ++y)
    {
        for (size_t x{ 0 }; x < width; x += 8)
        {
            if constexpr (std::is_integral_v<T>)
                accum += abs(to_float(load<T>(&srcp[x])) - to_float(load<T>(&srcp2[x])));
            else
                accum += abs(Vec8f().load(&srcp[x]) - Vec8f().load(&srcp2[x]));
        }

        srcp += pitch;
        srcp2 += pitch2;
    }

    if constexpr (std::is_integral_v<T>)
        return horizontal_add(accum / ((1 << bits_per_pixel) - 1)) / (height * width);
    else
        return horizontal_add(accum) / (height * width);
}

template float ComparePlane_avx2<uint8_t>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;
template float ComparePlane_avx2<uint16_t>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;
template float ComparePlane_avx2<float>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;

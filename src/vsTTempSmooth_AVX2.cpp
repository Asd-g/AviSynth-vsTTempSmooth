#include "VCL2/vectorclass.h"
#include "vsTTempSmooth.h"

short my_extract_epi16_from256(__m256i ymm_data, int idx)
{
    __m256i ymm_low8;
    if (idx > 7)
    {
        ymm_low8 = _mm256_permute2x128_si256(ymm_data, ymm_data, 1);
        idx -= 8;
    }
    else
        ymm_low8 = ymm_data;

    //	ymm_low8 = _mm256_srli_si256(ymm_low8, idx);

    __m128i xmm_low8 = _mm256_castsi256_si128(ymm_low8);

    switch (idx)
    {
        case 7:
            xmm_low8 = _mm_srli_si128(xmm_low8, 14);
            break;
        case 6:
            xmm_low8 = _mm_srli_si128(xmm_low8, 12);
            break;
        case 5:
            xmm_low8 = _mm_srli_si128(xmm_low8, 10);
            break;
        case 4:
            xmm_low8 = _mm_srli_si128(xmm_low8, 8);
            break;
        case 3:
            xmm_low8 = _mm_srli_si128(xmm_low8, 6);
            break;
        case 2:
            xmm_low8 = _mm_srli_si128(xmm_low8, 4);
            break;
        case 1:
            xmm_low8 = _mm_srli_si128(xmm_low8, 2);
            break;
    }

    int extr = _mm_cvtsi128_si32(xmm_low8);

    return (short)extr;
}

int my_extract_epi32_from256(__m256i ymm_data, int idx)
{
    __m256i ymm_low4;
    if (idx > 3)
    {
        ymm_low4 = _mm256_permute2x128_si256(ymm_data, ymm_data, 1);
        idx -= 4;
    }
    else
        ymm_low4 = ymm_data;

    __m128i xmm_low4 = _mm256_castsi256_si128(ymm_low4);

    switch (idx)
    {
        case 3:
            xmm_low4 = _mm_srli_si128(xmm_low4, 12);
            break;
        case 2:
            xmm_low4 = _mm_srli_si128(xmm_low4, 8);
            break;
        case 1:
            xmm_low4 = _mm_srli_si128(xmm_low4, 4);
            break;
    }

    return _mm_cvtsi128_si32(xmm_low4);
}


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
                        const auto check_v1_01{ diff01 < thresh&& abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh&& abs(t1_02 - t2_02) < thresh };

                        const auto& t2_03{ t1_03 };
                        t1_03 = load<T>(&pfp[frameIndex][x + 16]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{ diff03 < thresh&& abs(t1_03 - t2_03) < thresh };

                        const auto& t2_04{ t1_04 };
                        t1_04 = load<T>(&pfp[frameIndex][x + 24]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{ diff04 < thresh&& abs(t1_04 - t2_04) < thresh };

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
                        const auto check_v1_01{ diff01 < thresh&& abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh&& abs(t1_02 - t2_02) < thresh };

                        const auto& t2_03{ t1_03 };
                        t1_03 = load<T>(&pfp[frameIndex][x + 16]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{ diff03 < thresh&& abs(t1_03 - t2_03) < thresh };

                        const auto& t2_04{ t1_04 };
                        t1_04 = load<T>(&pfp[frameIndex][x + 24]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{ diff04 < thresh&& abs(t1_04 - t2_04) < thresh };

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
                        const auto check_v1_01{ diff01 < thresh&& abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh&& abs(t1_02 - t2_02) < thresh };

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
                        const auto check_v1_01{ diff01 < thresh&& abs(t1_01 - t2_01) < thresh };

                        const auto& t2_02{ t1_02 };
                        t1_02 = load<T>(&pfp[frameIndex][x + 8]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{ diff02 < thresh&& abs(t1_02 - t2_02) < thresh };

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
void TTempSmooth<pfclip, fp>::filterI_mode2_avx2_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{

    int DM_table[MAX_TEMP_RAD * 2 + 1][MAX_TEMP_RAD * 2 + 1];

    int src_stride[15]{};
    int pf_stride[15]{};
    const int stride{ dst->GetPitch(plane) };
    const int width{ dst->GetRowSize(plane) };
    const int height{ dst->GetHeight(plane) };
    const uint8_t* srcp[15]{}, * pfp[15]{};

    const int l{ plane >> 1 };
    const int thresh{ _thresh[l] << _shift };

    const int thUPD{ _thUPD[l] << _shift };
    const int pnew{ _pnew[l] << _shift };
    uint8_t* pMem;
    if ((plane >> 1) == 0) pMem = pIIRMemY;
    if ((plane >> 1) == 1) pMem = pIIRMemU;
    if ((plane >> 1) == 2) pMem = pIIRMemV;

    int* pMemSum;
    if ((plane >> 1) == 0) pMemSum = pMinSumMemY;
    if ((plane >> 1) == 1) pMemSum = pMinSumMemU;
    if ((plane >> 1) == 2) pMemSum = pMinSumMemV;

    const int iMaxSumDM = 255 * (_maxr * 2 + 1);

    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane);
        pf_stride[i] = pf[i]->GetPitch(plane);
        srcp[i] = reinterpret_cast<const uint8_t*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const uint8_t*>(pf[i]->GetReadPtr(plane));
    }

#ifdef _DEBUG
    iMEL_non_current_samples = 0;
    iMEL_mem_hits = 0;
#endif

    uint8_t* dstp{ reinterpret_cast<uint8_t*>(dst->GetWritePtr(plane)) };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 32)
        {
            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m256i ymm_row_l16;
            __m256i ymm_row_h16;
            __m256i ymm_col_l16;
            __m256i ymm_col_h16;
            __m256i ymm_zero = _mm256_setzero_si256();

            __m256i ymm_sum_minrow_l16 = _mm256_set1_epi16((short)iMaxSumDM); // hope 8bit 255-max diff with tr up to (2x7+1)=15 not overflow signed short ? max tr is 63 ? 
            __m256i ymm_sum_minrow_h16 = _mm256_set1_epi16((short)iMaxSumDM); // hope 8bit 255-max diff with tr up to (2x7+1)=15 not overflow signed short ? max tr is 63 ? 
            __m256i ymm_idx_minrow_l16 = _mm256_setzero_si256();
            __m256i ymm_idx_minrow_h16 = _mm256_setzero_si256();

            for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
            {
                __m256i ymm_sum_row_l16 = _mm256_setzero_si256();
                __m256i ymm_sum_row_h16 = _mm256_setzero_si256();

                for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                {
                    if (dmt_row == dmt_col)
                    { // block with itself => DM=0
                        continue;
                    }

                    // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                    uint8_t* row_data_ptr;
                    uint8_t* col_data_ptr;

                    if (dmt_row == _maxr) // src sample
                    {
                        row_data_ptr = (uint8_t*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        row_data_ptr = (uint8_t*)&srcp[dmt_row][x];
                    }

                    if (dmt_col == _maxr) // src sample
                    {
                        col_data_ptr = (uint8_t*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        col_data_ptr = (uint8_t*)&srcp[dmt_col][x];
                    }

                    __m256i ymm_row32 = _mm256_load_si256((const __m256i*)row_data_ptr);
                    __m256i ymm_col32 = _mm256_load_si256((const __m256i*)col_data_ptr);

                    ymm_row_l16 = _mm256_permute4x64_epi64(ymm_row32, 0x50);
                    ymm_row_h16 = _mm256_permute4x64_epi64(ymm_row32, 0xFA);

                    ymm_col_l16 = _mm256_permute4x64_epi64(ymm_col32, 0x50);
                    ymm_col_h16 = _mm256_permute4x64_epi64(ymm_col32, 0xFA);

                    ymm_row_l16 = _mm256_unpacklo_epi8(ymm_row_l16, ymm_zero);
                    ymm_row_h16 = _mm256_unpacklo_epi8(ymm_row_h16, ymm_zero);

                    ymm_col_l16 = _mm256_unpacklo_epi8(ymm_col_l16, ymm_zero);
                    ymm_col_h16 = _mm256_unpacklo_epi8(ymm_col_h16, ymm_zero);

                    __m256i ymm_subtr_l16 = _mm256_sub_epi16(ymm_row_l16, ymm_col_l16);
                    __m256i ymm_subtr_h16 = _mm256_sub_epi16(ymm_row_h16, ymm_col_h16);

                    __m256i ymm_abs_l16 = _mm256_abs_epi16(ymm_subtr_l16);
                    __m256i ymm_abs_h16 = _mm256_abs_epi16(ymm_subtr_h16);

                    ymm_sum_row_l16 = _mm256_add_epi16(ymm_sum_row_l16, ymm_abs_l16);
                    ymm_sum_row_h16 = _mm256_add_epi16(ymm_sum_row_h16, ymm_abs_h16);
                }

                __m256i ymm_mask_gt_l16 = _mm256_cmpgt_epi16(ymm_sum_minrow_l16, ymm_sum_row_l16);
                __m256i ymm_mask_gt_h16 = _mm256_cmpgt_epi16(ymm_sum_minrow_h16, ymm_sum_row_h16);

                __m256i ymm_idx_row = _mm256_set1_epi16((short)dmt_row);

                ymm_sum_minrow_l16 = _mm256_blendv_epi8(ymm_sum_minrow_l16, ymm_sum_row_l16, ymm_mask_gt_l16);
                ymm_sum_minrow_h16 = _mm256_blendv_epi8(ymm_sum_minrow_h16, ymm_sum_row_h16, ymm_mask_gt_h16);

                ymm_idx_minrow_l16 = _mm256_blendv_epi8(ymm_idx_minrow_l16, ymm_idx_row, ymm_mask_gt_l16);
                ymm_idx_minrow_h16 = _mm256_blendv_epi8(ymm_idx_minrow_h16, ymm_idx_row, ymm_mask_gt_h16);

            }

            for (int sub_x = 0; sub_x < 32; sub_x++)
            {
                int i_idx_minrow;
                int i_sum_minrow;

                if (sub_x < 16)
                {
                    i_idx_minrow = my_extract_epi16_from256(ymm_idx_minrow_l16, sub_x);
                    i_sum_minrow = my_extract_epi16_from256(ymm_sum_minrow_l16, sub_x);
                }
                else
                {
                    i_idx_minrow = my_extract_epi16_from256(ymm_idx_minrow_h16, sub_x - 16);
                    i_sum_minrow = my_extract_epi16_from256(ymm_sum_minrow_h16, sub_x - 16);
                }
#ifdef _DEBUG

                // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
                int i_sum_minrow_s = iMaxSumDM;
                int i_idx_minrow_s = 0;

                for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
                {
                    int i_sum_row_s = 0;
                    for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                    {
                        if (dmt_row == dmt_col)
                        { // block with itself => DM=0
                            continue;
                        }

                        // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                        uint8_t* row_data_ptr;
                        uint8_t* col_data_ptr;

                        if (dmt_row == _maxr) // src sample
                        {
                            row_data_ptr = (uint8_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            row_data_ptr = (uint8_t*)&srcp[dmt_row][x + sub_x];
                        }

                        if (dmt_col == _maxr) // src sample
                        {
                            col_data_ptr = (uint8_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            col_data_ptr = (uint8_t*)&srcp[dmt_col][x + sub_x];
                        }

                        i_sum_row_s += INTABS(*row_data_ptr - *col_data_ptr);
                    }

                    if (i_sum_row_s < i_sum_minrow_s)
                    {
                        i_sum_minrow_s = i_sum_row_s;
                        i_idx_minrow_s = dmt_row;
                    }
                }

                if (i_idx_minrow != i_idx_minrow_s)
                {
                    int idbr = 0;
                }

                if (i_sum_minrow != i_sum_minrow_s)
                {
                    int idbr = 0;
                }

#endif
                // set block of idx_minrow as output block
                const BYTE* best_data_ptr;

                if (i_idx_minrow == _maxr) // src sample
                {
                    best_data_ptr = &pfp[_maxr][x + sub_x];

                }
                else // ref sample
                {
                    best_data_ptr = &srcp[i_idx_minrow][x + sub_x];

#ifdef _DEBUG
                    iMEL_non_current_samples++;
#endif
                }

                if (thUPD > 0) // IIR here
                {
                    // IIR - check if memory sample is still good
                    int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);

                    if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) >= pMemSum[x + sub_x]))
                    {
                        //mem still good - output mem block
                        best_data_ptr = &pMem[x + sub_x];

#ifdef _DEBUG
                        iMEL_mem_hits++;
#endif
                    }
                    else // mem no good - update mem
                    {
                        pMem[x + sub_x] = *best_data_ptr;
                        pMemSum[x + sub_x] = i_sum_minrow;
                    }
                }

                // check if best is below thresh-difference from current
                if (INTABS(*best_data_ptr - srcp[_maxr][x + sub_x]) < thresh)
                {
                    dstp[x + sub_x] = *best_data_ptr;
                }
                else
                {
                    dstp[x + sub_x] = srcp[_maxr][x + sub_x];
                }
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

#ifdef _DEBUG
    float fRatioMEL_non_current_samples = (float)iMEL_non_current_samples / (float)(width * height);
    float fRatioMEL_mem_samples = (float)iMEL_mem_hits / (float)(width * height);
    int idbr = 0;
#endif
}

template void TTempSmooth<true, true>::filterI_mode2_avx2_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterI_mode2_avx2_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterI_mode2_avx2_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterI_mode2_avx2_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);



template<bool pfclip, bool fp>
void TTempSmooth<pfclip, fp>::filterI_mode2_avx2_g_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{

    int DM_table[MAX_TEMP_RAD * 2 + 1][MAX_TEMP_RAD * 2 + 1];
#define SIMD_AVX2_SPP 32
    int alignas(32) Temp[(MAX_TEMP_RAD * 2 + 1) * SIMD_AVX2_SPP + SIMD_AVX2_SPP * 2];

    // make temp ptr 32-bytes aligned
    int* pTemp = &Temp[0];


    int src_stride[15]{};
    int pf_stride[15]{};
    const int stride{ dst->GetPitch(plane) };
    const int width{ dst->GetRowSize(plane) };
    const int height{ dst->GetHeight(plane) };
    const uint8_t* srcp[15]{}, * pfp[15]{};

    const int l{ plane >> 1 };
    const int thresh{ _thresh[l] << _shift };

    const int thUPD{ _thUPD[l] << _shift };
    const int pnew{ _pnew[l] << _shift };
    uint8_t* pMem;
    if ((plane >> 1) == 0) pMem = pIIRMemY;
    if ((plane >> 1) == 1) pMem = pIIRMemU;
    if ((plane >> 1) == 2) pMem = pIIRMemV;

    int* pMemSum;
    if ((plane >> 1) == 0) pMemSum = pMinSumMemY;
    if ((plane >> 1) == 1) pMemSum = pMinSumMemU;
    if ((plane >> 1) == 2) pMemSum = pMinSumMemV;

    const int iMaxSumDM = 255 * (_maxr * 2 + 1);

    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane);
        pf_stride[i] = pf[i]->GetPitch(plane);
        srcp[i] = reinterpret_cast<const uint8_t*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const uint8_t*>(pf[i]->GetReadPtr(plane));
    }

#ifdef _DEBUG
    iMEL_non_current_samples = 0;
    iMEL_mem_hits = 0;
#endif

    uint8_t* dstp{ reinterpret_cast<uint8_t*>(dst->GetWritePtr(plane)) };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 32)
        {
            // copy all input frames processed samples in SIMD pass in the temp buf in uint32 form
            __m256i ymm_l8_1;
            __m256i ymm_l8_2;
            __m256i ymm_h8_1;
            __m256i ymm_h8_2;

            __m256i ymm_zero = _mm256_setzero_si256();
            __m256i ymm_idx_0_3_4_7 = _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 0);
            __m256i ymm_idx_8_11_12_15 = _mm256_set_epi32(0, 0, 0, 3, 0, 0, 0, 2);
            __m256i ymm_idx_16_19_20_23 = _mm256_set_epi32(0, 0, 0, 5, 0, 0, 0, 4);
            __m256i ymm_idx_24_27_28_31 = _mm256_set_epi32(0, 0, 0, 7, 0, 0, 0, 6);


            for (int i = 0; i < (_maxr * 2 + 1); i++)
            {
                uint8_t* data_ptr;
                if (i == _maxr) // src sample
                {
                    data_ptr = (uint8_t*)&pfp[_maxr][x];
                }
                else // ref sample
                {
                    data_ptr = (uint8_t*)&srcp[i][x];
                }

                __m256i ymm_src32 = _mm256_load_si256((const __m256i*)data_ptr);

                ymm_l8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_0_3_4_7);
                ymm_l8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_8_11_12_15);
                ymm_h8_1 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_16_19_20_23);
                ymm_h8_2 = _mm256_permutevar8x32_epi32(ymm_src32, ymm_idx_24_27_28_31);

                ymm_l8_1 = _mm256_unpacklo_epi8(ymm_l8_1, ymm_zero);
                ymm_l8_2 = _mm256_unpacklo_epi8(ymm_l8_2, ymm_zero);
                ymm_h8_1 = _mm256_unpacklo_epi8(ymm_h8_1, ymm_zero);
                ymm_h8_2 = _mm256_unpacklo_epi8(ymm_h8_2, ymm_zero);

                ymm_l8_1 = _mm256_unpacklo_epi16(ymm_l8_1, ymm_zero);
                ymm_l8_2 = _mm256_unpacklo_epi16(ymm_l8_2, ymm_zero);
                ymm_h8_1 = _mm256_unpacklo_epi16(ymm_h8_1, ymm_zero);
                ymm_h8_2 = _mm256_unpacklo_epi16(ymm_h8_2, ymm_zero);

                _mm256_store_si256((__m256i*)(pTemp + (8 * 4) * i), ymm_l8_1);
                _mm256_store_si256((__m256i*)(pTemp + (8 * 4) * i + 8), ymm_l8_2);
                _mm256_store_si256((__m256i*)(pTemp + (8 * 4) * i + 16), ymm_h8_1);
                _mm256_store_si256((__m256i*)(pTemp + (8 * 4) * i + 24), ymm_h8_2);

            }

            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m256i ymm_row_l8_1;
            __m256i ymm_row_l8_2;
            __m256i ymm_row_h8_1;
            __m256i ymm_row_h8_2;
            __m256i ymm_col_l8_1;
            __m256i ymm_col_l8_2;
            __m256i ymm_col_h8_1;
            __m256i ymm_col_h8_2;

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
                    { // block with itself => DM=0
                        continue;
                    }

                    int* row_data_ptr = &pTemp[dmt_row * (4 * 8)];
                    int* col_data_ptr = &pTemp[dmt_col * (4 * 8)];

                    ymm_row_l8_1 = _mm256_load_si256((const __m256i*)(row_data_ptr));
                    ymm_row_l8_2 = _mm256_load_si256((const __m256i*)(row_data_ptr + 8)); // int_ptr

                    ymm_row_h8_1 = _mm256_load_si256((const __m256i*)(row_data_ptr + 16));
                    ymm_row_h8_2 = _mm256_load_si256((const __m256i*)(row_data_ptr + 24));

                    ymm_col_l8_1 = _mm256_load_si256((const __m256i*)(col_data_ptr));
                    ymm_col_l8_2 = _mm256_load_si256((const __m256i*)(col_data_ptr + 8));

                    ymm_col_h8_1 = _mm256_load_si256((const __m256i*)(col_data_ptr + 16));
                    ymm_col_h8_2 = _mm256_load_si256((const __m256i*)(col_data_ptr + 24));


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

            __m256i ymm_best_data_l8_1;
            __m256i ymm_best_data_l8_2;
            __m256i ymm_best_data_h8_1;
            __m256i ymm_best_data_h8_2;

            __m256i ymm_idx_mul = _mm256_set1_epi32(SIMD_AVX2_SPP);

            ymm_idx_minrow_l8_1 = _mm256_mullo_epi32(ymm_idx_minrow_l8_1, ymm_idx_mul);
            ymm_idx_minrow_l8_2 = _mm256_mullo_epi32(ymm_idx_minrow_l8_2, ymm_idx_mul);
            ymm_idx_minrow_h8_1 = _mm256_mullo_epi32(ymm_idx_minrow_h8_1, ymm_idx_mul);
            ymm_idx_minrow_h8_2 = _mm256_mullo_epi32(ymm_idx_minrow_h8_2, ymm_idx_mul);

            ymm_best_data_l8_1 = _mm256_i32gather_epi32(pTemp, ymm_idx_minrow_l8_1, 4);
            ymm_best_data_l8_2 = _mm256_i32gather_epi32(pTemp, ymm_idx_minrow_l8_2, 4);
            ymm_best_data_h8_1 = _mm256_i32gather_epi32(pTemp, ymm_idx_minrow_h8_1, 4);
            ymm_best_data_h8_2 = _mm256_i32gather_epi32(pTemp, ymm_idx_minrow_h8_2, 4);

            ymm_best_data_l8_1 = _mm256_packus_epi32(ymm_best_data_l8_1, ymm_zero);
            ymm_best_data_l8_2 = _mm256_packus_epi32(ymm_best_data_l8_2, ymm_zero);
            ymm_best_data_h8_1 = _mm256_packus_epi32(ymm_best_data_h8_1, ymm_zero);
            ymm_best_data_h8_2 = _mm256_packus_epi32(ymm_best_data_h8_2, ymm_zero);

            ymm_best_data_l8_1 = _mm256_packus_epi16(ymm_best_data_l8_1, ymm_zero);
            ymm_best_data_l8_2 = _mm256_packus_epi16(ymm_best_data_l8_2, ymm_zero);
            ymm_best_data_h8_1 = _mm256_packus_epi16(ymm_best_data_h8_1, ymm_zero);
            ymm_best_data_h8_2 = _mm256_packus_epi16(ymm_best_data_h8_2, ymm_zero);

            __m256i ymm_idx_4_7 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 4, 0);
            __m256i ymm_idx_12_15 = _mm256_set_epi32(0, 0, 0, 0, 4, 0, 0, 0);
            __m256i ymm_idx_20_23 = _mm256_set_epi32(0, 0, 4, 0, 0, 0, 0, 0);
            __m256i ymm_idx_28_31 = _mm256_set_epi32(4, 0, 0, 0, 0, 0, 0, 0);

            ymm_best_data_l8_1 = _mm256_permutevar8x32_epi32(ymm_best_data_l8_1, ymm_idx_4_7);
            ymm_best_data_l8_2 = _mm256_permutevar8x32_epi32(ymm_best_data_l8_2, ymm_idx_12_15);
            ymm_best_data_h8_1 = _mm256_permutevar8x32_epi32(ymm_best_data_h8_1, ymm_idx_20_23);
            ymm_best_data_h8_2 = _mm256_permutevar8x32_epi32(ymm_best_data_h8_2, ymm_idx_28_31);

            __m256i ymm_out = _mm256_blend_epi32(ymm_best_data_l8_1, ymm_best_data_l8_2, 0x0C);
            ymm_out = _mm256_blend_epi32(ymm_out, ymm_best_data_h8_1, 0x30);
            ymm_out = _mm256_blend_epi32(ymm_out, ymm_best_data_h8_2, 0xC0);

            uint8_t* pDst = &dstp[x];
            _mm256_store_si256((__m256i*)(pDst), ymm_out);


            /*
            for (int sub_x = 0; sub_x < 32; sub_x++)
            {
                int i_idx_minrow;
                int i_sum_minrow;

                if (sub_x < 8)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_l8_1, sub_x);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_l8_1, sub_x);
                }
                else if (sub_x < 16)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_l8_2, sub_x - 8);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_l8_2, sub_x - 8);
                }
                else if (sub_x < 24)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_h8_1, sub_x - 16);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_h8_1, sub_x - 16);
                }
                else
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_h8_2, sub_x - 24);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_h8_2, sub_x - 24);
                }
#ifdef _DEBUG

                // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
                int i_sum_minrow_s = iMaxSumDM;
                int i_idx_minrow_s = 0;

                for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
                {
                    int i_sum_row_s = 0;
                    for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                    {
                        if (dmt_row == dmt_col)
                        { // block with itself => DM=0
                            continue;
                        }

                        // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                        uint8_t* row_data_ptr;
                        uint8_t* col_data_ptr;

                        if (dmt_row == _maxr) // src sample
                        {
                            row_data_ptr = (uint8_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            row_data_ptr = (uint8_t*)&srcp[dmt_row][x + sub_x];
                        }

                        if (dmt_col == _maxr) // src sample
                        {
                            col_data_ptr = (uint8_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            col_data_ptr = (uint8_t*)&srcp[dmt_col][x + sub_x];
                        }

                        i_sum_row_s += INTABS(*row_data_ptr - *col_data_ptr);
                    }

                    if (i_sum_row_s < i_sum_minrow_s)
                    {
                        i_sum_minrow_s = i_sum_row_s;
                        i_idx_minrow_s = dmt_row;
                    }
                }

                if (i_idx_minrow != i_idx_minrow_s)
                {
                    int idbr = 0;
                }

                if (i_sum_minrow != i_sum_minrow_s)
                {
                    int idbr = 0;
                }

#endif
                // set block of idx_minrow as output block
                const BYTE* best_data_ptr;

                if (i_idx_minrow == _maxr) // src sample
                {
                    best_data_ptr = &pfp[_maxr][x + sub_x];

                }
                else // ref sample
                {
                    best_data_ptr = &srcp[i_idx_minrow][x + sub_x];

#ifdef _DEBUG
                    iMEL_non_current_samples++;
#endif
                }

                if (thUPD > 0) // IIR here
                {
                    // IIR - check if memory sample is still good
                    int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);

                    if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) >= pMemSum[x + sub_x]))
                    {
                        //mem still good - output mem block
                        best_data_ptr = &pMem[x + sub_x];

#ifdef _DEBUG
                        iMEL_mem_hits++;
#endif
                    }
                    else // mem no good - update mem
                    {
                        pMem[x + sub_x] = *best_data_ptr;
                        pMemSum[x + sub_x] = i_sum_minrow;
                    }
                }

                // check if best is below thresh-difference from current
                if (INTABS(*best_data_ptr - srcp[_maxr][x + sub_x]) < thresh)
                {
                    dstp[x + sub_x] = *best_data_ptr;
                }
                else
                {
                    dstp[x + sub_x] = srcp[_maxr][x + sub_x];
                }
            } */

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

#ifdef _DEBUG
    float fRatioMEL_non_current_samples = (float)iMEL_non_current_samples / (float)(width * height);
    float fRatioMEL_mem_samples = (float)iMEL_mem_hits / (float)(width * height);
    int idbr = 0;
#endif
}

template void TTempSmooth<true, true>::filterI_mode2_avx2_g_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterI_mode2_avx2_g_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterI_mode2_avx2_g_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterI_mode2_avx2_g_uint8(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);



template<bool pfclip, bool fp>
void TTempSmooth<pfclip, fp>::filterI_mode2_avx2_uint16(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{

    int DM_table[MAX_TEMP_RAD * 2 + 1][MAX_TEMP_RAD * 2 + 1];

    int src_stride[15]{};
    int pf_stride[15]{};
    const int stride{ dst->GetPitch(plane) / 2 };
    const int width{ dst->GetRowSize(plane) / 2 }; // in samples 
    const int height{ dst->GetHeight(plane) };
    const uint16_t* srcp[15]{}, * pfp[15]{};

    const int l{ plane >> 1 };
    const int thresh{ _thresh[l] << _shift };

    const int thUPD{ _thUPD[l] << _shift };
    const int pnew{ _pnew[l] << _shift };
    uint16_t* pMem;
    if ((plane >> 1) == 0) pMem = reinterpret_cast<uint16_t*>(pIIRMemY);
    if ((plane >> 1) == 1) pMem = reinterpret_cast<uint16_t*>(pIIRMemU);
    if ((plane >> 1) == 2) pMem = reinterpret_cast<uint16_t*>(pIIRMemV);

    int* pMemSum;
    if ((plane >> 1) == 0) pMemSum = pMinSumMemY;
    if ((plane >> 1) == 1) pMemSum = pMinSumMemU;
    if ((plane >> 1) == 2) pMemSum = pMinSumMemV;

    const int iMaxSumDM = 65535 * (_maxr * 2 + 1);

    for (int i{ 0 }; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / 2;
        pf_stride[i] = pf[i]->GetPitch(plane) / 2;
        srcp[i] = reinterpret_cast<const uint16_t*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const uint16_t*>(pf[i]->GetReadPtr(plane));
    }

#ifdef _DEBUG
    iMEL_non_current_samples = 0;
    iMEL_mem_hits = 0;
#endif

    uint16_t* dstp{ reinterpret_cast<uint16_t*>(dst->GetWritePtr(plane)) };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 32)
        {
            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            __m256i ymm_row_l8_1;
            __m256i ymm_row_l8_2;
            __m256i ymm_row_h8_1;
            __m256i ymm_row_h8_2;
            __m256i ymm_col_l8_1;
            __m256i ymm_col_l8_2;
            __m256i ymm_col_h8_1;
            __m256i ymm_col_h8_2;
            __m256i ymm_zero = _mm256_setzero_si256();

            __m256i ymm_sum_minrow_l8_1 = _mm256_set1_epi32(iMaxSumDM); // hope 16bit 65535-max diff with tr up to (2x7+1)=15 not overflow signed int ? max tr is 63 ? 
            __m256i ymm_sum_minrow_l8_2 = _mm256_set1_epi32(iMaxSumDM); // 
            __m256i ymm_sum_minrow_h8_1 = _mm256_set1_epi32(iMaxSumDM); // 
            __m256i ymm_sum_minrow_h8_2 = _mm256_set1_epi32(iMaxSumDM); // 

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
                    { // block with itself => DM=0
                        continue;
                    }

                    // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                    uint16_t* row_data_ptr;
                    uint16_t* col_data_ptr;

                    if (dmt_row == _maxr) // src sample
                    {
                        row_data_ptr = (uint16_t*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        row_data_ptr = (uint16_t*)&srcp[dmt_row][x];
                    }

                    if (dmt_col == _maxr) // src sample
                    {
                        col_data_ptr = (uint16_t*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        col_data_ptr = (uint16_t*)&srcp[dmt_col][x];
                    }

                    __m256i ymm_row16_1 = _mm256_load_si256((const __m256i*)row_data_ptr);
                    __m256i ymm_row16_2 = _mm256_load_si256((const __m256i*)(row_data_ptr + 16)); // in shorts
                    __m256i ymm_col16_1 = _mm256_load_si256((const __m256i*)col_data_ptr);
                    __m256i ymm_col16_2 = _mm256_load_si256((const __m256i*)(col_data_ptr + 16)); // in shorts


                    ymm_row_l8_1 = _mm256_permute4x64_epi64(ymm_row16_1, 0x10);
                    ymm_row_l8_2 = _mm256_permute4x64_epi64(ymm_row16_1, 0x32);

                    ymm_row_h8_1 = _mm256_permute4x64_epi64(ymm_row16_2, 0x10);
                    ymm_row_h8_2 = _mm256_permute4x64_epi64(ymm_row16_2, 0x32);

                    ymm_col_l8_1 = _mm256_permute4x64_epi64(ymm_col16_1, 0x10);
                    ymm_col_l8_2 = _mm256_permute4x64_epi64(ymm_col16_1, 0x32);

                    ymm_col_h8_1 = _mm256_permute4x64_epi64(ymm_col16_2, 0x10);
                    ymm_col_h8_2 = _mm256_permute4x64_epi64(ymm_col16_2, 0x32);

                    ymm_row_l8_1 = _mm256_unpacklo_epi16(ymm_row_l8_1, ymm_zero);
                    ymm_row_l8_2 = _mm256_unpacklo_epi16(ymm_row_l8_2, ymm_zero);

                    ymm_row_h8_1 = _mm256_unpacklo_epi16(ymm_row_h8_1, ymm_zero);
                    ymm_row_h8_2 = _mm256_unpacklo_epi16(ymm_row_h8_2, ymm_zero);

                    ymm_col_l8_1 = _mm256_unpacklo_epi16(ymm_col_l8_1, ymm_zero);
                    ymm_col_l8_2 = _mm256_unpacklo_epi16(ymm_col_l8_2, ymm_zero);

                    ymm_col_h8_1 = _mm256_unpacklo_epi16(ymm_col_h8_1, ymm_zero);
                    ymm_col_h8_2 = _mm256_unpacklo_epi16(ymm_col_h8_2, ymm_zero);


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

            for (int sub_x = 0; sub_x < 32; sub_x++)
            {
                int i_idx_minrow;
                int i_sum_minrow;

                if (sub_x < 8)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_l8_1, sub_x);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_l8_1, sub_x);
                }
                else if (sub_x < 16)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_l8_2, sub_x - 8);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_l8_2, sub_x - 8);
                }
                else if (sub_x < 24)
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_h8_1, sub_x - 16);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_h8_1, sub_x - 16);
                }
                else
                {
                    i_idx_minrow = my_extract_epi32_from256(ymm_idx_minrow_h8_2, sub_x - 24);
                    i_sum_minrow = my_extract_epi32_from256(ymm_sum_minrow_h8_2, sub_x - 24);
                }

#ifdef _DEBUG

                // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
                int i_sum_minrow_s = iMaxSumDM;
                int i_idx_minrow_s = 0;

                for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
                {
                    int i_sum_row_s = 0;
                    for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                    {
                        if (dmt_row == dmt_col)
                        { // block with itself => DM=0
                            continue;
                        }

                        // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                        uint16_t* row_data_ptr;
                        uint16_t* col_data_ptr;

                        if (dmt_row == _maxr) // src sample
                        {
                            row_data_ptr = (uint16_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            row_data_ptr = (uint16_t*)&srcp[dmt_row][x + sub_x];
                        }

                        if (dmt_col == _maxr) // src sample
                        {
                            col_data_ptr = (uint16_t*)&pfp[_maxr][x + sub_x];
                        }
                        else // ref block
                        {
                            col_data_ptr = (uint16_t*)&srcp[dmt_col][x + sub_x];
                        }

                        i_sum_row_s += INTABS(*row_data_ptr - *col_data_ptr);
                    }

                    if (i_sum_row_s < i_sum_minrow_s)
                    {
                        i_sum_minrow_s = i_sum_row_s;
                        i_idx_minrow_s = dmt_row;
                    }
                }

                if (i_idx_minrow != i_idx_minrow_s)
                {
                    int idbr = 0;
                }

                if (i_sum_minrow != i_sum_minrow_s)
                {
                    int idbr = 0;
                }

#endif

                // set block of idx_minrow as output block
                const uint16_t* best_data_ptr;

                if (i_idx_minrow == _maxr) // src sample
                {
                    best_data_ptr = &pfp[_maxr][x + sub_x];

                }
                else // ref sample
                {
                    best_data_ptr = &srcp[i_idx_minrow][x + sub_x];

#ifdef _DEBUG
                    iMEL_non_current_samples++;
#endif
                }

                if (thUPD > 0) // IIR here
                {
                    // IIR - check if memory sample is still good
                    int idm_mem = INTABS(*best_data_ptr - pMem[x + sub_x]);

                    if ((idm_mem < thUPD) && ((i_sum_minrow + pnew) >= pMemSum[x + sub_x]))
                    {
                        //mem still good - output mem block
                        best_data_ptr = &pMem[x + sub_x];

#ifdef _DEBUG
                        iMEL_mem_hits++;
#endif
                    }
                    else // mem no good - update mem
                    {
                        pMem[x + sub_x] = *best_data_ptr;
                        pMemSum[x + sub_x] = i_sum_minrow;
                    }
                }

                // check if best is below thresh-difference from current
                if (INTABS(*best_data_ptr - srcp[_maxr][x + sub_x]) < thresh)
                {
                    dstp[x + sub_x] = *best_data_ptr;
                }
                else
                {
                    dstp[x + sub_x] = srcp[_maxr][x + sub_x];
                }
            }

        }

        for (int i{ 0 }; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
        pMem += width;// mem_stride in 16bit ??
        pMemSum += width;
    }

#ifdef _DEBUG
    float fRatioMEL_non_current_samples = (float)iMEL_non_current_samples / (float)(width * height);
    float fRatioMEL_mem_samples = (float)iMEL_mem_hits / (float)(width * height);
    int idbr = 0;
#endif
}

template void TTempSmooth<true, true>::filterI_mode2_avx2_uint16(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<true, false>::filterI_mode2_avx2_uint16(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, true>::filterI_mode2_avx2_uint16(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);
template void TTempSmooth<false, false>::filterI_mode2_avx2_uint16(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane);






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
                    const auto check_v1{ diff < thresh&& min(abs(t1 - t2), 1.0f) < thresh };

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
                    const auto check_v1{ diff < thresh&& min(abs(t1 - t2), 1.0f) < thresh };

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

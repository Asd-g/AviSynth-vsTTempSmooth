#include "VCL2/vectorclass.h"
#include "vsTTempSmooth.h"

template<typename T>
AVS_FORCEINLINE static Vec16i load(const void* p)
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return Vec16i().load_16uc(p);
    else
        return Vec16i().load_16us(p);
}

template<bool pfclip, bool fp>
template<typename T, bool useDiff>
void TTempSmooth<pfclip, fp>::filterI_avx512(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept
{
    int src_stride[15]{};
    int pf_stride[15]{};
    const size_t stride{dst->GetPitch(plane) / sizeof(T)};
    const int width{static_cast<int>(dst->GetRowSize(plane) / sizeof(T))};
    const int height{dst->GetHeight(plane)};
    const T *srcp[15]{}, *pfp[15]{};
    for (int i{0}; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / sizeof(T);
        pf_stride[i] = pf[i]->GetPitch(plane) / sizeof(T);
        srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
    }

    T* __restrict dstp{reinterpret_cast<T*>(dst->GetWritePtr(plane))};

    const int l{plane >> 1};
    const float* const weightSaved{_weight[l].data()};
    const Vec16i thresh{_thresh[l] << _shift};

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        for (int y{0}; y < height; ++y)
        {
            for (int x{0}; x < width; x += 64)
            {
                const auto& c01{load<T>(&pfp[_maxr][x])};
                const auto& srcp_v01{load<T>(&srcp[_maxr][x])};

                const auto& c02{load<T>(&pfp[_maxr][x + 16])};
                const auto& srcp_v02{load<T>(&srcp[_maxr][x + 16])};

                const auto& c03{load<T>(&pfp[_maxr][x + 32])};
                const auto& srcp_v03{load<T>(&srcp[_maxr][x + 32])};

                const auto& c04{load<T>(&pfp[_maxr][x + 48])};
                const auto& srcp_v04{load<T>(&srcp[_maxr][x + 48])};

                Vec16f weights01{_cw};
                auto sum01{to_float(srcp_v01) * weights01};

                Vec16f weights02{_cw};
                auto sum02{to_float(srcp_v02) * weights02};

                Vec16f weights03{_cw};
                auto sum03{to_float(srcp_v03) * weights03};

                Vec16f weights04{_cw};
                auto sum04{to_float(srcp_v04) * weights04};

                int frameIndex{_maxr - 1};

                if (frameIndex > fromFrame)
                {
                    auto t1_01{load<T>(&pfp[frameIndex][x])};
                    auto diff01{abs(c01 - t1_01)};
                    const auto check_v01{diff01 < thresh};

                    auto t1_02{load<T>(&pfp[frameIndex][x + 16])};
                    auto diff02{abs(c02 - t1_02)};
                    const auto check_v02{diff02 < thresh};

                    auto t1_03{load<T>(&pfp[frameIndex][x + 32])};
                    auto diff03{abs(c03 - t1_03)};
                    const auto check_v03{diff03 < thresh};

                    auto t1_04{load<T>(&pfp[frameIndex][x + 48])};
                    auto diff04{abs(c04 - t1_04)};
                    const auto check_v04{diff04 < thresh};

                    auto weight01{(useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight02{(useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight03{(useDiff) ? lookup<1792>(diff03 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight04{(useDiff) ? lookup<1792>(diff04 >> _shift, weightSaved) : weightSaved[frameIndex]};

                    weights01 = select(Vec16fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec16fb(check_v02), weights02 + weight02, weights02);
                    weights03 = select(Vec16fb(check_v03), weights03 + weight03, weights03);
                    weights04 = select(Vec16fb(check_v04), weights04 + weight04, weights04);

                    sum01 = select(Vec16fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec16fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);
                    sum03 = select(Vec16fb(check_v03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 32])), weight03, sum03), sum03);
                    sum04 = select(Vec16fb(check_v04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 48])), weight04, sum04), sum04);

                    --frameIndex;
                    int v{256};

                    while (frameIndex > fromFrame)
                    {
                        const auto& t2_01{t1_01};
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{diff01 < thresh && abs(t1_01 - t2_01) < thresh};

                        const auto& t2_02{t1_02};
                        t1_02 = load<T>(&pfp[frameIndex][x + 16]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{diff02 < thresh && abs(t1_02 - t2_02) < thresh};

                        const auto& t2_03{t1_03};
                        t1_03 = load<T>(&pfp[frameIndex][x + 32]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{diff03 < thresh && abs(t1_03 - t2_03) < thresh};

                        const auto& t2_04{t1_04};
                        t1_04 = load<T>(&pfp[frameIndex][x + 48]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{diff04 < thresh && abs(t1_04 - t2_04) < thresh};

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight03 = (useDiff) ? lookup<1792>((diff03 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight04 = (useDiff) ? lookup<1792>((diff04 >> _shift) + v, weightSaved) : weightSaved[frameIndex];

                        weights01 = select(Vec16fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec16fb(check_v1_02), weights02 + weight02, weights02);
                        weights03 = select(Vec16fb(check_v1_03), weights03 + weight03, weights03);
                        weights04 = select(Vec16fb(check_v1_04), weights04 + weight04, weights04);

                        sum01 = select(Vec16fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec16fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);
                        sum03 = select(Vec16fb(check_v1_03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 32])), weight03, sum03), sum03);
                        sum04 = select(Vec16fb(check_v1_04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 48])), weight04, sum04), sum04);

                        --frameIndex;
                        v += 256;
                    }
                }

                frameIndex = _maxr + 1;

                if (frameIndex < toFrame)
                {
                    auto t1_01{load<T>(&pfp[frameIndex][x])};
                    auto diff01{abs(c01 - t1_01)};
                    const auto check_v01{diff01 < thresh};

                    auto t1_02{load<T>(&pfp[frameIndex][x + 16])};
                    auto diff02{abs(c02 - t1_02)};
                    const auto check_v02{diff02 < thresh};

                    auto t1_03{load<T>(&pfp[frameIndex][x + 32])};
                    auto diff03{abs(c03 - t1_03)};
                    const auto check_v03{diff03 < thresh};

                    auto t1_04{load<T>(&pfp[frameIndex][x + 48])};
                    auto diff04{abs(c04 - t1_04)};
                    const auto check_v04{diff04 < thresh};

                    auto weight01{(useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight02{(useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight03{(useDiff) ? lookup<1792>(diff03 >> _shift, weightSaved) : weightSaved[frameIndex]};
                    auto weight04{(useDiff) ? lookup<1792>(diff04 >> _shift, weightSaved) : weightSaved[frameIndex]};

                    weights01 = select(Vec16fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec16fb(check_v02), weights02 + weight02, weights02);
                    weights03 = select(Vec16fb(check_v03), weights03 + weight03, weights03);
                    weights04 = select(Vec16fb(check_v04), weights04 + weight04, weights04);

                    sum01 = select(Vec16fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec16fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);
                    sum03 = select(Vec16fb(check_v03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 32])), weight03, sum03), sum03);
                    sum04 = select(Vec16fb(check_v04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 48])), weight04, sum04), sum04);

                    ++frameIndex;
                    int v{256};

                    while (frameIndex < toFrame)
                    {
                        const auto& t2_01{t1_01};
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{diff01 < thresh && abs(t1_01 - t2_01) < thresh};

                        const auto& t2_02{t1_02};
                        t1_02 = load<T>(&pfp[frameIndex][x + 16]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{diff02 < thresh && abs(t1_02 - t2_02) < thresh};

                        const auto& t2_03{t1_03};
                        t1_03 = load<T>(&pfp[frameIndex][x + 32]);
                        diff03 = abs(c03 - t1_03);
                        const auto check_v1_03{diff03 < thresh && abs(t1_03 - t2_03) < thresh};

                        const auto& t2_04{t1_04};
                        t1_04 = load<T>(&pfp[frameIndex][x + 48]);
                        diff04 = abs(c04 - t1_04);
                        const auto check_v1_04{diff04 < thresh && abs(t1_04 - t2_04) < thresh};

                        weight01 = (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight02 = (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight03 = (useDiff) ? lookup<1792>((diff03 >> _shift) + v, weightSaved) : weightSaved[frameIndex];
                        weight04 = (useDiff) ? lookup<1792>((diff04 >> _shift) + v, weightSaved) : weightSaved[frameIndex];

                        weights01 = select(Vec16fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec16fb(check_v1_02), weights02 + weight02, weights02);
                        weights03 = select(Vec16fb(check_v1_03), weights03 + weight03, weights03);
                        weights04 = select(Vec16fb(check_v1_04), weights04 + weight04, weights04);

                        sum01 = select(Vec16fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec16fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);
                        sum03 = select(Vec16fb(check_v1_03), mul_add(to_float(load<T>(&srcp[frameIndex][x + 32])), weight03, sum03), sum03);
                        sum04 = select(Vec16fb(check_v1_04), mul_add(to_float(load<T>(&srcp[frameIndex][x + 48])), weight04, sum04), sum04);

                        ++frameIndex;
                        v += 256;
                    }
                }

                if constexpr (fp)
                {
                    compress_saturated_s2u(
                        compress_saturated(
                            truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x])), (1.0f - weights01), sum01 + 0.5f)), zero_si512()),
                        zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + x);
                    compress_saturated_s2u(
                        compress_saturated(
                            truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 16])), (1.0f - weights02), sum02 + 0.5f)), zero_si512()),
                        zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 16));
                    compress_saturated_s2u(
                        compress_saturated(
                            truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 32])), (1.0f - weights03), sum03 + 0.5f)), zero_si512()),
                        zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 32));
                    compress_saturated_s2u(
                        compress_saturated(
                            truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 48])), (1.0f - weights04), sum04 + 0.5f)), zero_si512()),
                        zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 48));
                }
                else
                {
                    compress_saturated_s2u(compress_saturated(truncatei(sum01 / weights01 + 0.5f), zero_si512()), zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + x);
                    compress_saturated_s2u(compress_saturated(truncatei(sum02 / weights02 + 0.5f), zero_si512()), zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 16));
                    compress_saturated_s2u(compress_saturated(truncatei(sum03 / weights03 + 0.5f), zero_si512()), zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 32));
                    compress_saturated_s2u(compress_saturated(truncatei(sum04 / weights04 + 0.5f), zero_si512()), zero_si512())
                        .get_low()
                        .get_low()
                        .store(dstp + (x + 48));
                }
            }

            for (int i{0}; i < _diameter; ++i)
            {
                srcp[i] += src_stride[i];
                pfp[i] += pf_stride[i];
            }

            dstp += stride;
        }
    }
    else
    {
        for (int y{0}; y < height; ++y)
        {
            for (int x{0}; x < width; x += 32)
            {
                const auto& c01{load<T>(&pfp[_maxr][x])};
                const auto& srcp_v01{load<T>(&srcp[_maxr][x])};

                const auto& c02{load<T>(&pfp[_maxr][x + 16])};
                const auto& srcp_v02{load<T>(&srcp[_maxr][x + 16])};

                Vec16f weights01{_cw};
                auto sum01{to_float(srcp_v01) * weights01};

                Vec16f weights02{_cw};
                auto sum02{to_float(srcp_v02) * weights02};

                int frameIndex{_maxr - 1};

                if (frameIndex > fromFrame)
                {
                    auto t1_01{load<T>(&pfp[frameIndex][x])};
                    auto diff01{abs(c01 - t1_01)};
                    const auto check_v01{diff01 < thresh};

                    auto t1_02{load<T>(&pfp[frameIndex][x + 16])};
                    auto diff02{abs(c02 - t1_02)};
                    const auto check_v02{diff02 < thresh};

                    auto weight01{(useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};
                    auto weight02{(useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};

                    weights01 = select(Vec16fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec16fb(check_v02), weights02 + weight02, weights02);

                    sum01 = select(Vec16fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec16fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);

                    --frameIndex;
                    int v{256};

                    while (frameIndex > fromFrame)
                    {
                        const auto& t2_01{t1_01};
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{diff01 < thresh && abs(t1_01 - t2_01) < thresh};

                        const auto& t2_02{t1_02};
                        t1_02 = load<T>(&pfp[frameIndex][x + 16]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{diff02 < thresh && abs(t1_02 - t2_02) < thresh};

                        weight01 =
                            (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);
                        weight02 =
                            (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);

                        weights01 = select(Vec16fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec16fb(check_v1_02), weights02 + weight02, weights02);

                        sum01 = select(Vec16fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec16fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);

                        --frameIndex;
                        v += 256;
                    }
                }

                frameIndex = _maxr + 1;

                if (frameIndex < toFrame)
                {
                    auto t1_01{load<T>(&pfp[frameIndex][x])};
                    auto diff01{abs(c01 - t1_01)};
                    const auto check_v01{diff01 < thresh};

                    auto t1_02{load<T>(&pfp[frameIndex][x + 16])};
                    auto diff02{abs(c02 - t1_02)};
                    const auto check_v02{diff02 < thresh};

                    auto weight01{(useDiff) ? lookup<1792>(diff01 >> _shift, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};
                    auto weight02{(useDiff) ? lookup<1792>(diff02 >> _shift, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};

                    weights01 = select(Vec16fb(check_v01), weights01 + weight01, weights01);
                    weights02 = select(Vec16fb(check_v02), weights02 + weight02, weights02);

                    sum01 = select(Vec16fb(check_v01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                    sum02 = select(Vec16fb(check_v02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);

                    ++frameIndex;
                    int v{256};

                    while (frameIndex < toFrame)
                    {
                        const auto& t2_01{t1_01};
                        t1_01 = load<T>(&pfp[frameIndex][x]);
                        diff01 = abs(c01 - t1_01);
                        const auto check_v1_01{diff01 < thresh && abs(t1_01 - t2_01) < thresh};

                        const auto& t2_02{t1_02};
                        t1_02 = load<T>(&pfp[frameIndex][x + 16]);
                        diff02 = abs(c02 - t1_02);
                        const auto check_v1_02{diff02 < thresh && abs(t1_02 - t2_02) < thresh};

                        weight01 =
                            (useDiff) ? lookup<1792>((diff01 >> _shift) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);
                        weight02 =
                            (useDiff) ? lookup<1792>((diff02 >> _shift) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);

                        weights01 = select(Vec16fb(check_v1_01), weights01 + weight01, weights01);
                        weights02 = select(Vec16fb(check_v1_02), weights02 + weight02, weights02);

                        sum01 = select(Vec16fb(check_v1_01), mul_add(to_float(load<T>(&srcp[frameIndex][x])), weight01, sum01), sum01);
                        sum02 = select(Vec16fb(check_v1_02), mul_add(to_float(load<T>(&srcp[frameIndex][x + 16])), weight02, sum02), sum02);

                        ++frameIndex;
                        v += 256;
                    }
                }

                if constexpr (fp)
                {
                    compress_saturated_s2u(
                        truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x])), (1.0f - weights01), sum01 + 0.5f)), zero_si512())
                        .get_low()
                        .store(dstp + x);
                    compress_saturated_s2u(
                        truncatei(mul_add(to_float(load<T>(&srcp[_maxr][x + 16])), (1.0f - weights02), sum02 + 0.5f)), zero_si512())
                        .get_low()
                        .store(dstp + (x + 16));
                }
                else
                {
                    compress_saturated_s2u(truncatei(sum01 / weights01 + 0.5f), zero_si512()).get_low().store(dstp + x);
                    compress_saturated_s2u(truncatei(sum02 / weights02 + 0.5f), zero_si512()).get_low().store(dstp + (x + 16));
                }
            }

            for (int i{0}; i < _diameter; ++i)
            {
                srcp[i] += src_stride[i];
                pfp[i] += pf_stride[i];
            }

            dstp += stride;
        }
    }
}

template void TTempSmooth<true, true>::filterI_avx512<uint8_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx512<uint8_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterI_avx512<uint8_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx512<uint8_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterI_avx512<uint8_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx512<uint8_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterI_avx512<uint8_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx512<uint8_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<true, true>::filterI_avx512<uint16_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx512<uint16_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterI_avx512<uint16_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterI_avx512<uint16_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterI_avx512<uint16_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx512<uint16_t, true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterI_avx512<uint16_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterI_avx512<uint16_t, false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template<bool pfclip, bool fp>
template<bool useDiff>
void TTempSmooth<pfclip, fp>::filterF_avx512(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept
{
    int src_stride[15]{};
    int pf_stride[15]{};
    const int stride{dst->GetPitch(plane) / 4};
    const int width{dst->GetRowSize(plane) / 4};
    const int height{dst->GetHeight(plane)};
    const float *srcp[15]{}, *pfp[15]{};
    for (int i{0}; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / 4;
        pf_stride[i] = pf[i]->GetPitch(plane) / 4;
        srcp[i] = reinterpret_cast<const float*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const float*>(pf[i]->GetReadPtr(plane));
    }

    float* __restrict dstp{reinterpret_cast<float*>(dst->GetWritePtr(plane))};

    const int l{plane >> 1};
    const float* const weightSaved{_weight[l].data()};
    const Vec16f thresh{_threshF[l]};

    for (int y{0}; y < height; ++y)
    {
        for (int x{0}; x < width; x += 16)
        {
            const auto& c{Vec16f().load(&pfp[_maxr][x])};
            const auto& srcp_v{Vec16f().load(&srcp[_maxr][x])};

            Vec16f weights{_cw};
            auto sum{srcp_v * weights};

            int frameIndex{_maxr - 1};

            if (frameIndex > fromFrame)
            {
                auto t1{Vec16f().load(&pfp[frameIndex][x])};
                auto diff{min(abs(c - t1), 1.0f)};
                const auto check_v{diff < thresh};

                auto weight{
                    (useDiff) ? lookup<1792>(truncatei(diff * 255.0f), weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};
                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, mul_add(Vec16f().load(&srcp[frameIndex][x]), weight, sum), sum);

                --frameIndex;
                int v{256};

                while (frameIndex > fromFrame)
                {
                    const auto& t2{t1};
                    t1 = Vec16f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{diff < thresh && min(abs(t1 - t2), 1.0f) < thresh};

                    weight =
                        (useDiff) ? lookup<1792>(truncatei(diff * 255.0f) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);
                    weights = select(check_v1, weights + weight, weights);
                    sum = select(check_v1, mul_add(Vec16f().load(&srcp[frameIndex][x]), weight, sum), sum);

                    --frameIndex;
                    v += 256;
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                auto t1{Vec16f().load(&pfp[frameIndex][x])};
                auto diff{min(abs(c - t1), 1.0f)};
                const auto check_v{diff < thresh};

                auto weight{
                    (useDiff) ? lookup<1792>(truncatei(diff * 255.0f), weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved)};
                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, mul_add(Vec16f().load(&srcp[frameIndex][x]), weight, sum), sum);

                ++frameIndex;
                int v{256};

                while (frameIndex < toFrame)
                {
                    const auto& t2{t1};
                    t1 = Vec16f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{diff < thresh && min(abs(t1 - t2), 1.0f) < thresh};

                    weight =
                        (useDiff) ? lookup<1792>(truncatei(diff * 255.0f) + v, weightSaved) : lookup<1792>(Vec16i(frameIndex), weightSaved);
                    weights = select(check_v1, weights + weight, weights);
                    sum = select(check_v1, mul_add(Vec16f().load(&srcp[frameIndex][x]), weight, sum), sum);

                    ++frameIndex;
                    v += 256;
                }
            }

            if constexpr (fp)
                mul_add(Vec16f().load(&srcp[_maxr][x]), (1.0f - weights), sum).store(dstp + x);
            else
                (sum / weights).store(dstp + x);
        }

        for (int i{0}; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
    }
}

template void TTempSmooth<true, true>::filterF_avx512<true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterF_avx512<true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, true>::filterF_avx512<false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<true, false>::filterF_avx512<false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template void TTempSmooth<false, true>::filterF_avx512<true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterF_avx512<true>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, true>::filterF_avx512<false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;
template void TTempSmooth<false, false>::filterF_avx512<false>(
    PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane) noexcept;

template<typename T>
float ComparePlane_avx512(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept
{
    const size_t pitch{src->GetPitch(PLANAR_Y) / sizeof(T)};
    const size_t pitch2{src1->GetPitch(PLANAR_Y) / sizeof(T)};
    const int width{static_cast<int>(src->GetRowSize(PLANAR_Y) / sizeof(T))};
    const int height{src->GetHeight(PLANAR_Y)};
    const T* srcp{reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y))};
    const T* srcp2{reinterpret_cast<const T*>(src1->GetReadPtr(PLANAR_Y))};

    Vec16f accum{0.0f};

    for (int y{0}; y < height; ++y)
    {
        for (int x{0}; x < width; x += 16)
        {
            if constexpr (std::is_integral_v<T>)
                accum += abs(to_float(load<T>(&srcp[x])) - to_float(load<T>(&srcp2[x])));
            else
                accum += abs(Vec16f().load(&srcp[x]) - Vec16f().load(&srcp2[x]));
        }

        srcp += pitch;
        srcp2 += pitch2;
    }

    if constexpr (std::is_integral_v<T>)
        return horizontal_add(accum / ((1 << bits_per_pixel) - 1)) / (height * width);
    else
        return horizontal_add(accum) / (height * width);
}

template float ComparePlane_avx512<uint8_t>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;
template float ComparePlane_avx512<uint16_t>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;
template float ComparePlane_avx512<float>(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept;

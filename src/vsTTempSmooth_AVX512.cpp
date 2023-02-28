#include "VCL2/vectorclass.h"
#include "vsTTempSmooth.h"


template <typename T>
AVS_FORCEINLINE static Vec16i load(const void* p)
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return Vec16i().load_16uc(p);
    else
        return Vec16i().load_16us(p);
}

template <bool pfclip>
template <typename T, bool useDiff, bool fp>
void TTempSmooth<pfclip>::filterI_avx512(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept
{
    src_stride /= sizeof(T);
    const size_t stride{ dst->GetPitch(plane) / sizeof(T) };
    const size_t width{ dst->GetRowSize(plane) / sizeof(T) };
    const int height{ dst->GetHeight(plane) };
    const T* srcp[15] = {}, * pfp[15] = {};
    for (int i{ 0 }; i < _diameter; ++i)
    {
        srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
    }

    T* __restrict dstp{ reinterpret_cast<T*>(dst->GetWritePtr(plane)) };

    const int l{ plane >> 1 };
    const Vec16i thresh{ _thresh[l] << _shift };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 16)
        {
            const auto& c{ load<T>(&pfp[_maxr][x]) };
            const auto& srcp_v{ load<T>(&srcp[_maxr][x]) };

            Vec16f weights{ _cw };
            Vec16f sum{ to_float(srcp_v) * weights };

            int frameIndex{ _maxr - 1 };

            if (frameIndex > fromFrame)
            {
                auto t1{ load<T>(&pfp[frameIndex][x]) };
                auto diff{ abs(c - t1) };
                const auto check_v{ diff < thresh };

                Vec16f weight;
                for (int i{ 0 }; i < 16; ++i)
                    weight.insert(i, _weight[l][(useDiff) ? (diff.extract(i) >> _shift) : frameIndex]);

                weights = select(Vec16fb(check_v), weights + weight, weights);
                sum = select(Vec16fb(check_v), sum + to_float(load<T>(&srcp[frameIndex][x])) * weight, sum);

                --frameIndex;
                int v{ 256 };

                while (frameIndex > fromFrame)
                {
                    const auto& t2{ t1 };
                    t1 = load<T>(&pfp[frameIndex][x]);
                    diff = abs(c - t1);
                    const auto check_v1{ diff < thresh&& abs(t1 - t2) < thresh };

                    Vec16f weight;
                    for (int i{ 0 }; i < 16; ++i)
                        weight.insert(i, _weight[l][(useDiff) ? ((diff.extract(i) >> _shift) + v) : frameIndex]);

                    weights = select(Vec16fb(check_v1), weights + weight, weights);
                    sum = select(Vec16fb(check_v1), sum + to_float(load<T>(&srcp[frameIndex][x])) * weight, sum);

                    --frameIndex;
                    v += 256;
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                auto t1{ load<T>(&pfp[frameIndex][x]) };
                auto diff{ abs(c - t1) };
                const auto check_v{ diff < thresh };

                Vec16f weight;
                for (int i{ 0 }; i < 16; ++i)
                    weight.insert(i, _weight[l][(useDiff) ? (diff.extract(i) >> _shift) : frameIndex]);

                weights = select(Vec16fb(check_v), weights + weight, weights);
                sum = select(Vec16fb(check_v), sum + to_float(load<T>(&srcp[frameIndex][x])) * weight, sum);

                ++frameIndex;
                int v{ 256 };

                while (frameIndex < toFrame)
                {
                    const auto& t2{ t1 };
                    t1 = load<T>(&pfp[frameIndex][x]);
                    diff = abs(c - t1);
                    const auto check_v1{ diff < thresh&& abs(t1 - t2) < thresh };

                    Vec16f weight;
                    for (int i{ 0 }; i < 16; ++i)
                        weight.insert(i, _weight[l][(useDiff) ? ((diff.extract(i) >> _shift) + v) : frameIndex]);

                    weights = select(Vec16fb(check_v1), weights + weight, weights);
                    sum = select(Vec16fb(check_v1), sum + to_float(load<T>(&srcp[frameIndex][x])) * weight, sum);

                    ++frameIndex;
                    v += 256;
                }
            }

            if constexpr (std::is_same_v<T, uint8_t>)
            {
                if constexpr (fp)
                    compress_saturated_s2u(compress_saturated(truncatei(to_float(load<T>(&srcp[_maxr][x])) * (1.0f - weights) + sum + 0.5f), zero_si512()), zero_si512()).store(dstp + x);
                else
                    compress_saturated_s2u(compress_saturated(truncatei(sum / weights + 0.5f), zero_si512()), zero_si512()).store(dstp + x);
            }
            else
            {
                if constexpr (fp)
                    compress_saturated_s2u(truncatei(to_float(load<T>(&srcp[_maxr][x])) * (1.0f - weights) + sum + 0.5f), zero_si512()).store(dstp + x);
                else
                    compress_saturated_s2u(truncatei(sum / weights + 0.5f), zero_si512()).store(dstp + x);
            }
        }

        for (int i{ 0 }; i < _diameter; ++i)
        {
            srcp[i] += src_stride;
            pfp[i] += src_stride;
        }
        dstp += stride;
    }
}

template void TTempSmooth<1>::filterI_avx512<uint8_t, true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint8_t, true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint8_t, false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint8_t, false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template void TTempSmooth<0>::filterI_avx512<uint8_t, true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint8_t, true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint8_t, false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint8_t, false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template void TTempSmooth<1>::filterI_avx512<uint16_t, true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint16_t, true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint16_t, false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterI_avx512<uint16_t, false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template void TTempSmooth<0>::filterI_avx512<uint16_t, true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint16_t, true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint16_t, false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterI_avx512<uint16_t, false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template <bool pfclip>
template <bool useDiff, bool fp>
void TTempSmooth<pfclip>::filterF_avx512(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept
{
    src_stride /= 4;
    const int stride{ dst->GetPitch(plane) / 4 };
    const int width{ dst->GetRowSize(plane) / 4 };
    const int height{ dst->GetHeight(plane) };
    const float* srcp[15] = {}, * pfp[15] = {};
    for (int i{ 0 }; i < _diameter; ++i)
    {
        srcp[i] = reinterpret_cast<const float*>(src[i]->GetReadPtr(plane));
        pfp[i] = reinterpret_cast<const float*>(pf[i]->GetReadPtr(plane));
    }

    float* __restrict dstp{ reinterpret_cast<float*>(dst->GetWritePtr(plane)) };

    const int l{ plane >> 1 };
    const Vec16f thresh{ _threshF[l] };

    for (int y{ 0 }; y < height; ++y)
    {
        for (int x{ 0 }; x < width; x += 16)
        {
            const auto& c{ Vec16f().load(&pfp[_maxr][x]) };
            const auto& srcp_v{ Vec16f().load(&srcp[_maxr][x]) };

            Vec16f weights{ _cw };
            Vec16f sum{ srcp_v * weights };

            int frameIndex{ _maxr - 1 };

            if (frameIndex > fromFrame)
            {
                auto t1{ Vec16f().load(&pfp[frameIndex][x]) };
                auto diff{ min(abs(c - t1), 1.0f) };
                const auto check_v{ diff < thresh };

                Vec16f weight;
                for (int i{ 0 }; i < 16; ++i)
                    weight.insert(i, _weight[l][(useDiff) ? static_cast<int>(diff.extract(i) * 255.0f) : frameIndex]);

                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, sum + Vec16f().load(&srcp[frameIndex][x]) * weight, sum);

                --frameIndex;
                int v{ 256 };

                while (frameIndex > fromFrame)
                {
                    const auto& t2{ t1 };
                    t1 = Vec16f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{ diff < thresh&& min(abs(t1 - t2), 1.0f) < thresh };

                    Vec16f weight;
                    for (int i{ 0 }; i < 16; ++i)
                        weight.insert(i, _weight[l][(useDiff) ? (static_cast<int>(diff.extract(i) * 255.0f) + v) : frameIndex]);

                    weights = select(Vec16fb(check_v1), weights + weight, weights);
                    sum = select(Vec16fb(check_v1), sum + Vec16f().load(&srcp[frameIndex][x]) * weight, sum);

                    --frameIndex;
                    v += 256;
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                auto t1{ Vec16f().load(&pfp[frameIndex][x]) };
                auto diff{ min(abs(c - t1), 1.0f) };
                const auto check_v{ diff < thresh };

                Vec16f weight;
                for (int i{ 0 }; i < 16; ++i)
                    weight.insert(i, _weight[l][(useDiff) ? static_cast<int>(diff.extract(i) * 255.0f) : frameIndex]);

                weights = select(check_v, weights + weight, weights);
                sum = select(check_v, sum + Vec16f().load(&srcp[frameIndex][x]) * weight, sum);

                ++frameIndex;
                int v{ 256 };

                while (frameIndex < toFrame)
                {
                    const auto& t2{ t1 };
                    t1 = Vec16f().load(&pfp[frameIndex][x]);
                    diff = min(abs(c - t1), 1.0f);
                    const auto check_v1{ diff < thresh&& min(abs(t1 - t2), 1.0f) < thresh };

                    Vec16f weight;
                    for (int i{ 0 }; i < 16; ++i)
                        weight.insert(i, _weight[l][(useDiff) ? (static_cast<int>(diff.extract(i) * 255.0f) + v) : frameIndex]);

                    weights = select(Vec16fb(check_v1), weights + weight, weights);
                    sum = select(Vec16fb(check_v1), sum + Vec16f().load(&srcp[frameIndex][x]) * weight, sum);

                    ++frameIndex;
                    v += 256;
                }
            }

            if constexpr (fp)
                (Vec16f().load(&srcp[_maxr][x]) * (1.0f - weights) + sum).store_nt(dstp + x);
            else
                (sum / weights).store_nt(dstp + x);
        }

        for (int i{ 0 }; i < _diameter; ++i)
        {
            srcp[i] += src_stride;
            pfp[i] += src_stride;
        }
        dstp += stride;
    }
}

template void TTempSmooth<1>::filterF_avx512<true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterF_avx512<true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterF_avx512<false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<1>::filterF_avx512<false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template void TTempSmooth<0>::filterF_avx512<true, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterF_avx512<true, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterF_avx512<false, true>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;
template void TTempSmooth<0>::filterF_avx512<false, false>(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride) noexcept;

template <typename T>
float ComparePlane_avx512(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept
{
    const size_t pitch{ src->GetPitch(PLANAR_Y) / sizeof(T) };
    const size_t pitch2{ src1->GetPitch(PLANAR_Y) / sizeof(T) };
    const size_t width{ src->GetRowSize(PLANAR_Y) / sizeof(T) };
    const int height{ src->GetHeight(PLANAR_Y) };
    const T* srcp{ reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y)) };
    const T* srcp2{ reinterpret_cast<const T*>(src1->GetReadPtr(PLANAR_Y)) };

    Vec16f accum{ 0.0f };

    for (size_t y{ 0 }; y < height; ++y)
    {
        for (size_t x{ 0 }; x < width; x += 16)
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

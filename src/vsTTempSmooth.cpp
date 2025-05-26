#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

#include "vsTTempSmooth.h"

static std::vector<int64_t> getPascalRow(int k)
{
    if (k < 0)
        return {1LL};
    std::vector<int64_t> row(k + 1);
    if (k == 0)
    {
        row[0] = 1LL;
        return row;
    }
    row[0] = 1LL;
    for (int i = 1; i <= k; ++i)
    {
        for (int j = i; j >= 1; --j)
            row[j] += row[j - 1];
    }
    return row;
}

template<bool pfclip, bool fp_template_param>
template<typename T, bool useDiff>
void TTempSmooth<pfclip, fp_template_param>::filterI(
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
    const int thresh_val{_thresh[l] << _shift};
    const float* const weightSaved{_weight[l].data()};

    for (int y{0}; y < height; ++y)
    {
        for (int x{0}; x < width; ++x)
        {
            const int c{static_cast<int>(pfp[_maxr][x])};
            float current_weights{_cw};
            float current_sum{srcp[_maxr][x] * _cw};

            int frameIndex{_maxr - 1};

            if (frameIndex > fromFrame)
            {
                int t1{static_cast<int>(pfp[frameIndex][x])};
                int diff{std::abs(c - t1)};

                if (diff < thresh_val)
                {
                    int dist_from_center_1_based{_maxr - frameIndex};
                    int v_offset_base{256 * (dist_from_center_1_based - 1)};
                    float weight_val{weightSaved[useDiff ? ((diff >> _shift) + v_offset_base) : frameIndex]};
                    current_weights += weight_val;
                    current_sum += srcp[frameIndex][x] * weight_val;

                    --frameIndex;

                    while (frameIndex > fromFrame)
                    {
                        const int t2{t1};
                        t1 = pfp[frameIndex][x];
                        diff = std::abs(c - t1);
                        dist_from_center_1_based = _maxr - frameIndex;
                        v_offset_base = 256 * (dist_from_center_1_based - 1);

                        if (diff < thresh_val && std::abs(t1 - t2) < thresh_val)
                        {
                            weight_val = weightSaved[useDiff ? ((diff >> _shift) + v_offset_base) : frameIndex];
                            current_weights += weight_val;
                            current_sum += srcp[frameIndex][x] * weight_val;

                            --frameIndex;
                        }
                        else
                            break;
                    }
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                int t1{static_cast<int>(pfp[frameIndex][x])};
                int diff{std::abs(c - t1)};

                if (diff < thresh_val)
                {
                    int dist_from_center_1_based{frameIndex - _maxr};
                    int v_offset_base{256 * (dist_from_center_1_based - 1)};
                    float weight_val{weightSaved[useDiff ? ((diff >> _shift) + v_offset_base) : frameIndex]};
                    current_weights += weight_val;
                    current_sum += srcp[frameIndex][x] * weight_val;

                    ++frameIndex;

                    while (frameIndex < toFrame)
                    {
                        const int t2{t1};
                        t1 = pfp[frameIndex][x];
                        diff = std::abs(c - t1);
                        dist_from_center_1_based = frameIndex - _maxr;
                        v_offset_base = 256 * (dist_from_center_1_based - 1);

                        if (diff < thresh_val && std::abs(t1 - t2) < thresh_val)
                        {
                            weight_val = weightSaved[useDiff ? ((diff >> _shift) + v_offset_base) : frameIndex];
                            current_weights += weight_val;
                            current_sum += srcp[frameIndex][x] * weight_val;

                            ++frameIndex;
                        }
                        else
                            break;
                    }
                }
            }

            if constexpr (fp_template_param)
                dstp[x] = static_cast<T>(srcp[_maxr][x] * (1.f - current_weights) + current_sum + 0.5f);
            else
                dstp[x] = (current_weights == 0.0f) ? srcp[_maxr][x] : static_cast<T>(current_sum / current_weights + 0.5f);
        }

        for (int i{0}; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
    }
}

template<bool pfclip, bool fp_template_param>
template<bool useDiff>
void TTempSmooth<pfclip, fp_template_param>::filterF(
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
    const float thresh_val{_threshF[l]};
    const float* const weightSaved{_weight[l].data()};

    for (int y{0}; y < height; ++y)
    {
        for (int x{0}; x < width; ++x)
        {
            const float c{pfp[_maxr][x]};
            float current_weights{_cw};
            float current_sum{srcp[_maxr][x] * _cw};

            int frameIndex{_maxr - 1};

            if (frameIndex > fromFrame)
            {
                float t1{pfp[frameIndex][x]};
                float diff{std::min(std::abs(c - t1), 1.f)};

                if (diff < thresh_val)
                {
                    int dist_from_center_1_based{_maxr - frameIndex};
                    int v_offset_base{256 * (dist_from_center_1_based - 1)};
                    float weight{weightSaved[useDiff ? (static_cast<int>(diff * 255.f) + v_offset_base) : frameIndex]};
                    current_weights += weight;
                    current_sum += srcp[frameIndex][x] * weight;

                    --frameIndex;

                    while (frameIndex > fromFrame)
                    {
                        const float t2{t1};
                        t1 = pfp[frameIndex][x];
                        diff = std::min(std::abs(c - t1), 1.f);
                        dist_from_center_1_based = _maxr - frameIndex;
                        v_offset_base = 256 * (dist_from_center_1_based - 1);

                        if (diff < thresh_val && std::min(std::abs(t1 - t2), 1.f) < thresh_val)
                        {
                            weight = weightSaved[useDiff ? (static_cast<int>(diff * 255.f) + v_offset_base) : frameIndex];
                            current_weights += weight;
                            current_sum += srcp[frameIndex][x] * weight;

                            --frameIndex;
                        }
                        else
                            break;
                    }
                }
            }

            frameIndex = _maxr + 1;

            if (frameIndex < toFrame)
            {
                float t1{pfp[frameIndex][x]};
                float diff{std::min(std::abs(c - t1), 1.f)};

                if (diff < thresh_val)
                {
                    int dist_from_center_1_based = frameIndex - _maxr;
                    int v_offset_base = 256 * (dist_from_center_1_based - 1);
                    float weight{weightSaved[useDiff ? (static_cast<int>(diff * 255.f) + v_offset_base) : frameIndex]};
                    current_weights += weight;
                    current_sum += srcp[frameIndex][x] * weight;

                    ++frameIndex;

                    while (frameIndex < toFrame)
                    {
                        const float t2{t1};
                        t1 = pfp[frameIndex][x];
                        diff = std::min(std::abs(c - t1), 1.f);
                        dist_from_center_1_based = frameIndex - _maxr;
                        v_offset_base = 256 * (dist_from_center_1_based - 1);

                        if (diff < thresh_val && std::min(std::abs(t1 - t2), 1.f) < thresh_val)
                        {
                            weight = weightSaved[useDiff ? (static_cast<int>(diff * 255.f) + v_offset_base) : frameIndex];
                            current_weights += weight;
                            current_sum += srcp[frameIndex][x] * weight;

                            ++frameIndex;
                        }
                        else
                            break;
                    }
                }
            }

            if constexpr (fp_template_param)
                dstp[x] = srcp[_maxr][x] * (1.f - current_weights) + current_sum;
            else
                dstp[x] = (current_weights == 0.f) ? srcp[_maxr][x] : (current_sum / current_weights);
        }

        for (int i{0}; i < _diameter; ++i)
        {
            srcp[i] += src_stride[i];
            pfp[i] += pf_stride[i];
        }

        dstp += stride;
    }
}

template<bool pfclip, bool fp_template_param>
template<typename T>
void TTempSmooth<pfclip, fp_template_param>::filter_mode2_C(PVideoFrame src[(MAX_TEMP_RAD * 2 + 1)], PVideoFrame pf[(MAX_TEMP_RAD * 2 + 1)],
    PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane)
{
    int src_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    int pf_stride[(MAX_TEMP_RAD * 2 + 1)]{};
    const size_t stride{dst->GetPitch(plane) / sizeof(T)};
    const int width{static_cast<int>(dst->GetRowSize(plane) / sizeof(T))};
    const int height{dst->GetHeight(plane)};
    const T *g_srcp[(MAX_TEMP_RAD * 2 + 1)]{}, *g_pfp[(MAX_TEMP_RAD * 2 + 1)]{};

    const int l{plane >> 1};

    typedef typename std::conditional<sizeof(T) <= 2, int, float>::type working_t;

    const working_t thresh = (sizeof(T) <= 2) ? (_thresh[l] << _shift) : (_thresh[l] / 256.0f);

    const working_t thUPD = (sizeof(T) <= 2) ? (_thUPD[l] << _shift) : (_thUPD[l] / 256.0f);
    const working_t pnew = (sizeof(T) <= 2) ? (_pnew[l] << _shift) : (_pnew[l] / 256.0f);
    T* g_pMem{reinterpret_cast<T*>(pIIRMem[l].data())};
    working_t* g_pMemSum{reinterpret_cast<working_t*>(pMinSumMem[l].data())};
    const working_t MaxSumDM = (sizeof(T) < 2) ? 255 * (_maxr * 2 + 1) : 65535 * (_maxr * 2 + 1); // 65535 is enough max for float too

    for (int i{0}; i < _diameter; ++i)
    {
        src_stride[i] = src[i]->GetPitch(plane) / sizeof(T);
        pf_stride[i] = pf[i]->GetPitch(plane) / sizeof(T);
        g_srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
        g_pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
    }

    T* g_dstp{reinterpret_cast<T*>(dst->GetWritePtr(plane))};

#ifdef _DEBUG
    iMEL_non_current_samples = 0;
    iMEL_mem_hits = 0;
#endif

#pragma omp parallel for num_threads(_threads)
    for (int y = 0; y < height; ++y)
    {
        // local threads ptrs
        const T *srcp[(MAX_TEMP_RAD * 2 + 1)]{}, *pfp[(MAX_TEMP_RAD * 2 + 1)]{};
        T *dstp, *pMem;
        working_t* pMemSum;

        for (int i{0}; i < _diameter; ++i)
        {
            srcp[i] = g_srcp[i] + y * src_stride[i];
            pfp[i] = g_pfp[i] + y * pf_stride[i];
        }

        dstp = g_dstp + y * stride;
        pMem = g_pMem + y * width;
        pMemSum = g_pMemSum + y * width;

        for (int x{0}; x < width; ++x)
        {

            // find lowest sum of row in DM_table and index of row in single DM scan with DM calc
            working_t wt_sum_minrow = MaxSumDM;
            int i_idx_minrow = 0;

            for (int dmt_row = 0; dmt_row < (_maxr * 2 + 1); dmt_row++)
            {
                working_t wt_sum_row = 0;
                for (int dmt_col = 0; dmt_col < (_maxr * 2 + 1); dmt_col++)
                {
                    if (dmt_row == dmt_col)
                    { // block with itself => DM=0
                        continue;
                    }

                    // _maxr is current sample, 0,1,2... is -maxr, ... +maxr
                    T* row_data_ptr;
                    T* col_data_ptr;

                    if (dmt_row == _maxr) // src sample
                    {
                        row_data_ptr = (T*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        row_data_ptr = (T*)&srcp[dmt_row][x];
                    }

                    if (dmt_col == _maxr) // src sample
                    {
                        col_data_ptr = (T*)&pfp[_maxr][x];
                    }
                    else // ref block
                    {
                        col_data_ptr = (T*)&srcp[dmt_col][x];
                    }

                    wt_sum_row += (sizeof(T) <= 2) ? std::abs(*row_data_ptr - *col_data_ptr) : std::abs(*row_data_ptr - *col_data_ptr);
                }

                if (wt_sum_row < wt_sum_minrow)
                {
                    wt_sum_minrow = wt_sum_row;
                    i_idx_minrow = dmt_row;
                }
            }

            // set block of idx_minrow as output block
            const T* best_data_ptr;

            if (i_idx_minrow == _maxr) // src sample
            {
                best_data_ptr = &pfp[_maxr][x];
            }
            else // ref sample
            {
                best_data_ptr = &srcp[i_idx_minrow][x];

#ifdef _DEBUG
                iMEL_non_current_samples++;
#endif
            }

            if (thUPD > 0) // IIR here
            {
                // IIR - check if memory sample is still good
                working_t idm_mem = (sizeof(T) <= 2) ? std::abs(*best_data_ptr - pMem[x]) : std::abs(*best_data_ptr - pMem[x]);

                if ((idm_mem < thUPD) && ((wt_sum_minrow + pnew) > pMemSum[x]))
                {
                    // mem still good - output mem block
                    best_data_ptr = &pMem[x];

#ifdef _DEBUG
                    iMEL_mem_hits++;
#endif
                }
                else // mem no good - update mem
                {
                    pMem[x] = *best_data_ptr;
                    pMemSum[x] = wt_sum_minrow;
                }
            }

            // check if best is below thresh-difference from current src
            if (((sizeof(T) <= 2) ? std::abs(*best_data_ptr - pfp[_maxr][x]) : std::abs(*best_data_ptr - pfp[_maxr][x])) < thresh)
            {
                dstp[x] = *best_data_ptr;
            }
            else
            {
                dstp[x] = pfp[_maxr][x];
            }
        }
    }

#ifdef _DEBUG
    float fRatioMEL_non_current_samples = (float)iMEL_non_current_samples / (float)(width * height);
    float fRatioMEL_mem_samples = (float)iMEL_mem_hits / (float)(width * height);
    int idbr = 0;
#endif
}

template<typename pixel_t>
AVS_FORCEINLINE static float get_sad_c(
    const pixel_t* c_plane, const pixel_t* t_plane, size_t height, size_t width, size_t c_pitch, size_t t_pitch) noexcept
{
    float accum{0.0f};

    for (size_t y{0}; y < height; ++y)
    {
        for (size_t x{0}; x < width; ++x)
            accum += std::abs(t_plane[x] - c_plane[x]);

        c_plane += c_pitch;
        t_plane += t_pitch;
    }

    return accum;
}

template<typename T>
static float ComparePlane(PVideoFrame& src, PVideoFrame& src1, const int bits_per_pixel) noexcept
{
    const size_t pitch{src->GetPitch(PLANAR_Y) / sizeof(T)};
    const size_t pitch2{src1->GetPitch(PLANAR_Y) / sizeof(T)};
    const size_t width{src->GetRowSize(PLANAR_Y) / sizeof(T)};
    const int height{src->GetHeight(PLANAR_Y)};
    const T* srcp{reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y))};
    const T* srcp2{reinterpret_cast<const T*>(src1->GetReadPtr(PLANAR_Y))};

    const float sad{get_sad_c<T>(srcp, srcp2, height, width, pitch, pitch2)};

    float f{sad / (height * width)};

    if constexpr (std::is_integral_v<T>)
        f /= ((1 << bits_per_pixel) - 1);

    return f;
}

template<bool pfclip, bool fp_template_param>
TTempSmooth<pfclip, fp_template_param>::TTempSmooth(PClip _child, int maxr, int ythresh, int uthresh, int vthresh, int ymdiff, int umdiff,
    int vmdiff, int strength, float scthresh, int y, int u, int v, PClip pfclip_, int opt, int pmode, int ythupd, int uthupd, int vthupd,
    int ypnew, int upnew, int vpnew, int threads, IScriptEnvironment* env)
    : GenericVideoFilter(_child),
      _maxr(maxr),
      _scthresh(scthresh),
      _diameter(maxr * 2 + 1),
      _thresh{ythresh, uthresh, vthresh},
      _mdiff{ymdiff, umdiff, vmdiff},
      _shift(vi.BitsPerComponent() - 8),
      _threshF{0.0f, 0.0f, 0.0f},
      _cw(0.0f),
      _pfclip(pfclip_),
      _opt(opt),
      _pmode(pmode),
      _thUPD{ythupd, uthupd, vthupd},
      _pnew{ypnew, upnew, vpnew},
      _threads{threads}
{
    has_at_least_v8 = env->FunctionExists("propShow");

    if (vi.IsRGB() || !vi.IsPlanar())
        env->ThrowError("vsTTempSmooth: clip must be Y/YUV(A) 8..32-bit planar format.");
    if (_pmode == 0 || _pmode == 2)
    {
        if (_maxr < 1 || _maxr > 7)
            env->ThrowError("vsTTempSmooth: maxr must be between 1..7 for pmode=0 and pmode=2.");
    }
    else if (_pmode == 1)
    {
        if (_maxr < 1 || _maxr > MAX_TEMP_RAD)
            env->ThrowError("vsTTempSmooth: maxr must be between 1..%d.", MAX_TEMP_RAD);
    }
    else
        env->ThrowError("vsTTempSmooth: pmode must be 0, 1, or 2.");
    if (ythresh < 1 || ythresh > 256)
        env->ThrowError("vsTTempSmooth: ythresh must be between 1..256.");
    if (uthresh < 1 || uthresh > 256)
        env->ThrowError("vsTTempSmooth: uthresh must be between 1..256.");
    if (vthresh < 1 || vthresh > 256)
        env->ThrowError("vsTTempSmooth: vthresh must be between 1..256.");
    if (ymdiff < 0 || ymdiff > 255)
        env->ThrowError("vsTTempSmooth: ymdiff must be between 0..255.");
    if (umdiff < 0 || umdiff > 255)
        env->ThrowError("vsTTempSmooth: umdiff must be between 0..255.");
    if (vmdiff < 0 || vmdiff > 255)
        env->ThrowError("vsTTempSmooth: vmdiff must be between 0..255.");
    if (strength < 1 || strength > 8)
        env->ThrowError("vsTTempSmooth: strength must be between 1..8.");
    if (_scthresh < 0.f || _scthresh > 100.f)
        env->ThrowError("vsTTempSmooth: scthresh must be between 0.0..100.0.");
    if (_opt < -1 || _opt > 3)
        env->ThrowError("vsTTempSmooth: opt must be between -1..3.");
    if (ythupd < 0)
        env->ThrowError("vsTTempSmooth: ythupd must be greater than 0.");
    if (uthupd < 0)
        env->ThrowError("vsTTempSmooth: uthupd must be greater than 0.");
    if (vthupd < 0)
        env->ThrowError("vsTTempSmooth: vthupd must be greater than 0.");
    if (ypnew < 0)
        env->ThrowError("vsTTempSmooth: ypnew must be greater than 0.");
    if (upnew < 0)
        env->ThrowError("vsTTempSmooth: upnew must be greater than 0.");
    if (vpnew < 0)
        env->ThrowError("vsTTempSmooth: vpnew must be greater than 0.");

    const int thr{static_cast<int>(std::thread::hardware_concurrency())};

    if (_threads == 0)
        _threads = thr;
    else if (_threads < 0 || _threads > thr)
        env->ThrowError("vsTTempSmooth: threads must be between 0..%s.", std::to_string(thr).c_str());

    const bool avx512{!!(env->GetCPUFlags() & CPUF_AVX512F) && (opt < 0 || opt == 3)};
    const bool avx2{!!(env->GetCPUFlags() & CPUF_AVX2) && (opt < 0 || opt == 2)};
    const bool sse2{!!(env->GetCPUFlags() & CPUF_SSE2) && (opt < 0 || opt == 1)};

    if (!avx512 && opt == 3)
        env->ThrowError("FFTSpectrum: opt=3 requires AVX512.");
    if (!avx2 && opt == 2)
        env->ThrowError("FFTSpectrum: opt=2 requires AVX2.");
    if (!sse2 && opt == 1)
        env->ThrowError("FFTSpectrum: opt=1 requires SSE2.");

    if constexpr (pfclip)
    {
        const VideoInfo& vi1 = pfclip_->GetVideoInfo();
        if (!vi.IsSameColorspace(vi1) || vi.width != vi1.width || vi.height != vi1.height)
            env->ThrowError("vsTTempSmooth: pfclip must have the same dimension as the main clip and be the same format.");
        if (vi.num_frames != vi1.num_frames)
            env->ThrowError("vsTTempSmooth: pfclip's number of frames doesn't match.");
    }

    const int planes[3]{y, u, v};
    static constexpr int iMaxSum{std::numeric_limits<int>::max()};
    static constexpr float fMaxSum{std::numeric_limits<float>::max()};

    for (int i{0}; i < std::min(vi.NumComponents(), 3); ++i)
    {
        switch (planes[i])
        {
        case 3:
            proccesplanes[i] = 3;
            break;
        case 2:
            proccesplanes[i] = 2;
            break;
        case 1:
            proccesplanes[i] = 1;
            break;
        default:
            env->ThrowError("vsTTempSmooth: y / u / v must be between 1..3.");
        }

        if (proccesplanes[i] == 3) // not support maxr > 7 ?
        {
            if (_pmode == 0 || _pmode == 2)
            {
                std::vector<float> dt_final_dist_weights(_maxr + 1);

                if (_pmode == 2)
                {
                    if (strength <= 1)
                    {
                        const int k_binom{2 * _maxr};
                        std::vector<int64_t> coeffs{getPascalRow(k_binom)};
                        for (int d{0}; d <= _maxr; ++d)
                            dt_final_dist_weights[d] = static_cast<float>(coeffs[_maxr - d]);
                    }
                    else
                    {
                        const int plateau_edge_dist{strength - 1};
                        if (plateau_edge_dist >= _maxr)
                        {
                            for (int d{0}; d <= _maxr; ++d)
                                dt_final_dist_weights[d] = 1.0f;
                        }
                        else
                        {
                            const int num_points_falloff_one_side{_maxr - plateau_edge_dist + 1};
                            const int k_falloff_shape{2 * (num_points_falloff_one_side - 1)};
                            std::vector<int64_t> falloff_coeffs{getPascalRow(k_falloff_shape)};

                            const float peak_falloff_val{(k_falloff_shape / 2 < falloff_coeffs.size())
                                                             ? static_cast<float>(falloff_coeffs[k_falloff_shape / 2])
                                                             : 1.0f};

                            for (int d{0}; d <= _maxr; ++d)
                            {
                                if (d <= plateau_edge_dist)
                                    dt_final_dist_weights[d] = peak_falloff_val;
                                else
                                {
                                    const int dist_from_falloff_peak{d - plateau_edge_dist};
                                    const int coeff_idx{(k_falloff_shape / 2) - dist_from_falloff_peak};
                                    if (coeff_idx >= 0 && coeff_idx < falloff_coeffs.size())
                                        dt_final_dist_weights[d] = static_cast<float>(falloff_coeffs[coeff_idx]);
                                    else                                 // Should not happen with correct k_falloff_shape
                                        dt_final_dist_weights[d] = 0.0f; // Safety
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (int d{0}; d <= _maxr; ++d)
                    {
                        if (d < strength)
                            dt_final_dist_weights[d] = 1.0f;
                        else
                            dt_final_dist_weights[d] = 1.0f / (static_cast<float>(d - strength + 2));
                    }
                }

                float current_sum_of_dist_weights{dt_final_dist_weights[0]};
                for (int d{1}; d <= _maxr; ++d)
                    current_sum_of_dist_weights += dt_final_dist_weights[d] * 2.0f;
                if (current_sum_of_dist_weights == 0.0f)
                    current_sum_of_dist_weights = 1.0f;

                if (_thresh[i] > _mdiff[i] + 1)
                {
                    _weight[i].resize(256 * _maxr);
                    _cw = dt_final_dist_weights[0] / current_sum_of_dist_weights;

                    float rt[256]{};
                    const float step_val{256.0f / (_thresh[i] - std::min(_mdiff[i], _thresh[i] - 1))};
                    float base_val{256.0f};
                    for (int j{0}; j < _thresh[i]; ++j)
                    {
                        if (_mdiff[i] > j)
                            rt[j] = 256.f;
                        else
                        {
                            if (base_val > 0.f)
                                rt[j] = base_val;
                            else
                                break;
                            base_val -= step_val;
                        }
                    }
                    for (int j{1}; j <= _maxr; ++j)
                    {
                        const float normalized_dist_w{dt_final_dist_weights[j] / current_sum_of_dist_weights};
                        for (int v{0}; v < 256; ++v)
                            _weight[i][256 * (j - 1) + v] = normalized_dist_w * rt[v] / 256.0f;
                    }
                }
                else
                {
                    _weight[i].resize(_diameter);
                    _weight[i][_maxr] = dt_final_dist_weights[0] / current_sum_of_dist_weights;
                    for (int d{1}; d <= _maxr; ++d)
                    {
                        const float norm_w{dt_final_dist_weights[d] / current_sum_of_dist_weights};
                        _weight[i][_maxr - d] = norm_w;
                        _weight[i][_maxr + d] = norm_w;
                    }

                    _cw = _weight[i][_maxr];
                }

                if (vi.ComponentSize() == 4)
                    _threshF[i] = _thresh[i] / 256.f;
            }
            else if (_pmode == 1 && _thUPD[i] > 0)
            {
                const size_t num_elements_minsum{static_cast<size_t>(vi.width) * vi.height};
                pIIRMem[i].resize(num_elements_minsum * vi.ComponentSize(), 0);
                pMinSumMem[i].resize(num_elements_minsum, (vi.ComponentSize() < 4) ? iMaxSum : fMaxSum);
            }
        }
    }

    _opt = (!avx512) ? (!avx2) ? (!sse2) ? 0 : 1 : 2 : 3;

    if (_pmode == 1)
    {
        if (!(_opt == 0 || _opt == 2 || _opt == 3))
            env->ThrowError("vsTTempSmooth: pmode=1 requires opt=0, opt=2 or opt=3.");
        if (_opt == 3 && vi.ComponentSize() < 4)
            env->ThrowError("vsTTempSmooth: pmode=1 opt=3 supports only 32-bit bit depth.");
    }

    if (_opt == 3)
    {
        switch (vi.ComponentSize())
        {
        case 1:
            compare = ComparePlane_avx512<uint8_t>;
            break;
        case 2:
            compare = ComparePlane_avx512<uint16_t>;
            break;
        default: {
            compare = ComparePlane_avx512<float>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filterF_mode2_avx512;
        }
        }
    }
    else if (_opt == 2)
    {
        switch (vi.ComponentSize())
        {
        case 1: {
            compare = ComparePlane_avx2<uint8_t>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filterI_mode2_avx2<uint8_t>;
            break;
        }
        case 2: {
            compare = ComparePlane_avx2<uint16_t>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filterI_mode2_avx2<uint16_t>;
            break;
        }
        default: {
            compare = ComparePlane_avx2<float>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filterF_mode2_avx2;
        }
        }
    }
    else if (_opt == 1)
    {
        switch (vi.ComponentSize())
        {
        case 1:
            compare = ComparePlane_sse2<uint8_t>;
            break;
        case 2:
            compare = ComparePlane_sse2<uint16_t>;
            break;
        default:
            compare = ComparePlane_sse2<float>;
        }
    }
    else
    {
        switch (vi.ComponentSize())
        {
        case 1: {
            compare = ComparePlane<uint8_t>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filter_mode2_C<uint8_t>;
            break;
        }
        case 2: {
            compare = ComparePlane<uint16_t>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filter_mode2_C<uint16_t>;
            break;
        }
        default: {
            compare = ComparePlane<float>;
            if (_pmode == 1)
                filter_mode2_fn_ptr = &TTempSmooth::filter_mode2_C<float>;
        }
        }
    }

#ifdef _DEBUG
    iMEL_non_current_samples = 0;
    iMEL_mem_hits = 0;
    iMEL_mem_updates = 0;
#endif
}

template<bool pfclip, bool fp_template_param>
PVideoFrame __stdcall TTempSmooth<pfclip, fp_template_param>::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src[MAX_TEMP_RAD * 2 + 1]{};
    PVideoFrame pf[MAX_TEMP_RAD * 2 + 1]{};

    for (int i{n - _maxr}; i <= n + _maxr; ++i)
    {
        const int frameNumber{std::clamp(i, 0, vi.num_frames - 1)};

        src[i - n + _maxr] = child->GetFrame(frameNumber, env);

        if constexpr (pfclip)
            pf[i - n + _maxr] = _pfclip->GetFrame(frameNumber, env);
    }

    PVideoFrame dst{(has_at_least_v8) ? env->NewVideoFrameP(vi, &src[_maxr]) : env->NewVideoFrame(vi)};

    int fromFrame{-1};
    int toFrame{_diameter};
    const int bits_per_pixel{vi.BitsPerComponent()};

    if (_scthresh)
    {
        for (int i{_maxr}; i > 0; --i)
        {
            if (compare((pfclip) ? pf[i - 1] : src[i - 1], (pfclip) ? pf[i] : src[i], bits_per_pixel) > _scthresh / 100.f)
            {
                fromFrame = i;
                break;
            }
        }

        for (int i{_maxr}; i < _diameter - 1; ++i)
        {
            if (compare((pfclip) ? pf[i] : src[i], (pfclip) ? pf[i + 1] : src[i + 1], bits_per_pixel) > _scthresh / 100.f)
            {
                toFrame = i;
                break;
            }
        }
    }

    constexpr int planes_y[3]{PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int i{0}; i < std::min(vi.NumComponents(), 3); ++i)
    {
        if (proccesplanes[i] == 3)
        {
            if (_pmode == 1)
            {
                (this->*filter_mode2_fn_ptr)(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                continue;
            }

            const bool use_diff_for_filter{_thresh[i] > _mdiff[i] + 1};

            if (_opt == 3)
            {
                switch (vi.ComponentSize())
                {
                case 1: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_avx512<uint8_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_avx512<uint8_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                case 2: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_avx512<uint16_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_avx512<uint16_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                default: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterF_avx512<true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterF_avx512<false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                }
                }
            }
            else if (_opt == 2)
            {
                switch (vi.ComponentSize())
                {
                case 1: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_avx2<uint8_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_avx2<uint8_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                case 2: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_avx2<uint16_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_avx2<uint16_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                default: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterF_avx2<true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterF_avx2<false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                }
                }
            }
            else if (_opt == 1)
            {
                switch (vi.ComponentSize())
                {
                case 1: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_sse2<uint8_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_sse2<uint8_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                case 2: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI_sse2<uint16_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI_sse2<uint16_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                default: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterF_sse2<true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterF_sse2<false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                }
                }
            }
            else
            {
                switch (vi.ComponentSize())
                {
                case 1: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI<uint8_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI<uint8_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                case 2: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterI<uint16_t, true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterI<uint16_t, false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    break;
                }
                default: {
                    if (use_diff_for_filter)
                        TTempSmooth::filterF<true>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                    else
                        TTempSmooth::filterF<false>(src, (pfclip) ? pf : src, dst, fromFrame, toFrame, planes_y[i]);
                }
                }
            }
        }
        else if (proccesplanes[i] == 2)
            env->BitBlt(dst->GetWritePtr(planes_y[i]), dst->GetPitch(planes_y[i]), src[_maxr]->GetReadPtr(planes_y[i]),
                src[_maxr]->GetPitch(planes_y[i]), src[_maxr]->GetRowSize(planes_y[i]), src[_maxr]->GetHeight(planes_y[i]));
    }

    return dst;
}

AVSValue __cdecl Create_TTempSmooth(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    enum
    {
        Clip,
        Maxr,
        Ythresh,
        Uthresh,
        Vthresh,
        Ymdiff,
        Umdiff,
        Vmdiff,
        Strength,
        Scthresh,
        Fp,
        Y,
        U,
        V,
        Pfclip,
        Opt,
        Pmode,
        YthUPD,
        UthUPD,
        VthUPD,
        Ypnew,
        Upnew,
        Vpnew,
        Threads
    };

    PClip pfclip{(args[Pfclip].Defined() ? args[Pfclip].AsClip() : nullptr)};
    const bool fp_script_arg{args[Fp].AsBool(true)};

    if (pfclip)
    {
        if (fp_script_arg)
            return new TTempSmooth<true, true>(args[Clip].AsClip(), args[Maxr].AsInt(3), args[Ythresh].AsInt(4), args[Uthresh].AsInt(5),
                args[Vthresh].AsInt(5), args[Ymdiff].AsInt(2), args[Umdiff].AsInt(3), args[Vmdiff].AsInt(3), args[Strength].AsInt(2),
                args[Scthresh].AsFloatf(12), args[Y].AsInt(3), args[U].AsInt(3), args[V].AsInt(3), pfclip, args[Opt].AsInt(-1),
                args[Pmode].AsInt(0), args[YthUPD].AsInt(0), args[UthUPD].AsInt(0), args[VthUPD].AsInt(0), args[Ypnew].AsInt(0),
                args[Upnew].AsInt(0), args[Vpnew].AsInt(0), args[Threads].AsInt(1), env);
        else
            return new TTempSmooth<true, false>(args[Clip].AsClip(), args[Maxr].AsInt(3), args[Ythresh].AsInt(4), args[Uthresh].AsInt(5),
                args[Vthresh].AsInt(5), args[Ymdiff].AsInt(2), args[Umdiff].AsInt(3), args[Vmdiff].AsInt(3), args[Strength].AsInt(2),
                args[Scthresh].AsFloatf(12), args[Y].AsInt(3), args[U].AsInt(3), args[V].AsInt(3), pfclip, args[Opt].AsInt(-1),
                args[Pmode].AsInt(0), args[YthUPD].AsInt(0), args[UthUPD].AsInt(0), args[VthUPD].AsInt(0), args[Ypnew].AsInt(0),
                args[Upnew].AsInt(0), args[Vpnew].AsInt(0), args[Threads].AsInt(1), env);
    }
    else
    {
        if (fp_script_arg)
            return new TTempSmooth<false, true>(args[Clip].AsClip(), args[Maxr].AsInt(3), args[Ythresh].AsInt(4), args[Uthresh].AsInt(5),
                args[Vthresh].AsInt(5), args[Ymdiff].AsInt(2), args[Umdiff].AsInt(3), args[Vmdiff].AsInt(3), args[Strength].AsInt(2),
                args[Scthresh].AsFloatf(12), args[Y].AsInt(3), args[U].AsInt(3), args[V].AsInt(3), pfclip, args[Opt].AsInt(-1),
                args[Pmode].AsInt(0), args[YthUPD].AsInt(0), args[UthUPD].AsInt(0), args[VthUPD].AsInt(0), args[Ypnew].AsInt(0),
                args[Upnew].AsInt(0), args[Vpnew].AsInt(0), args[Threads].AsInt(1), env);
        else
            return new TTempSmooth<false, false>(args[Clip].AsClip(), args[Maxr].AsInt(3), args[Ythresh].AsInt(4), args[Uthresh].AsInt(5),
                args[Vthresh].AsInt(5), args[Ymdiff].AsInt(2), args[Umdiff].AsInt(3), args[Vmdiff].AsInt(3), args[Strength].AsInt(2),
                args[Scthresh].AsFloatf(12), args[Y].AsInt(3), args[U].AsInt(3), args[V].AsInt(3), pfclip, args[Opt].AsInt(-1),
                args[Pmode].AsInt(0), args[YthUPD].AsInt(0), args[UthUPD].AsInt(0), args[VthUPD].AsInt(0), args[Ypnew].AsInt(0),
                args[Upnew].AsInt(0), args[Vpnew].AsInt(0), args[Threads].AsInt(1), env);
    }
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("vsTTempSmooth",
        "c[maxr]i[ythresh]i[uthresh]i[vthresh]i[ymdiff]i[umdiff]i[vmdiff]i[strength]i[scthresh]f[fp]b[y]i[u]i[v]i[pfclip]c[opt]i[pmode]i["
        "ythupd]i[uthupd]i[vthupd]i[ypnew]i[upnew]i[vpnew]i[threads]i",
        Create_TTempSmooth, 0);
    return "vsTTempSmooth";
}

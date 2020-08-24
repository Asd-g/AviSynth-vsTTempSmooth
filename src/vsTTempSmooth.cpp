#include <inttypes.h>
#include <cmath>
#include <limits>

#include "avisynth.h"
#include "avs/minmax.h"

#define VS_RESTRICT __restrict

class TTempSmooth : public GenericVideoFilter
{
	int _maxr;
	int _ythresh, _uthresh, _vthresh;
	int _ymdiff, _umdiff, _vmdiff;
	float _scthresh;
	bool _fp;
	int _diameter, _thresh[3], _mdiff[3], _shift;
	float _threshF[3], * _weight[3], _cw;
	bool _y, _u, _v;
	bool proccesplanes[3];
	PClip _pfclip;
	bool has_at_least_v8;
	int planecount;

	template<typename T, bool useDiff>
	void filterI(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride, const TTempSmooth* const VS_RESTRICT, IScriptEnvironment* env) noexcept;
	template<bool useDiff>
	void filterF(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride, const TTempSmooth* const VS_RESTRICT, IScriptEnvironment* env) noexcept;

public:
	TTempSmooth(PClip _child, int maxr, int ythresh, int uthresh, int vthresh, int ymdiff, int umdiff, int vmdiff, int strength, float scthresh, bool fp, bool y, bool u, bool v, PClip pfclip, IScriptEnvironment* env);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	int __stdcall SetCacheHints(int cachehints, int frame_range)
	{
		return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
	}
	~TTempSmooth();
};

template<typename T, bool useDiff>
void TTempSmooth::filterI(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride, const TTempSmooth* const VS_RESTRICT, IScriptEnvironment* env) noexcept
{
	const int width = dst->GetRowSize(plane) / vi.ComponentSize();
	const int height = dst->GetHeight(plane);
	src_stride /= sizeof(T);
	const int stride = dst->GetPitch(plane) / sizeof(T);
	const T* srcp[15] = {}, * pfp[15] = {};
	for (int i = 0; i < _diameter; i++)
	{
		srcp[i] = reinterpret_cast<const T*>(src[i]->GetReadPtr(plane));
		pfp[i] = reinterpret_cast<const T*>(pf[i]->GetReadPtr(plane));
	}
	T* VS_RESTRICT dstp = reinterpret_cast<T*>(dst->GetWritePtr(plane));

	int l;
	if (plane == 1)
		l = 0;
	else if (plane == 2)
		l = 1;
	else
		l = 2;

	const int thresh = _thresh[l] << _shift;
	const float* const weightSaved = _weight[l];

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const int c = pfp[_maxr][x];
			float weights = _cw;
			float sum = srcp[_maxr][x] * _cw;

			int frameIndex = _maxr - 1;

			if (frameIndex > fromFrame)
			{
				int t1 = pfp[frameIndex][x];
				int diff = std::abs(c - t1);

				if (diff < thresh) {
					float weight = weightSaved[useDiff ? diff >> _shift : frameIndex];
					weights += weight;
					sum += srcp[frameIndex][x] * weight;

					frameIndex--;
					int v = 256;

					while (frameIndex > fromFrame)
					{
						const int t2 = t1;
						t1 = pfp[frameIndex][x];
						diff = std::abs(c - t1);

						if (diff < thresh && std::abs(t1 - t2) < thresh)
						{
							weight = weightSaved[useDiff ? (diff >> _shift) + v : frameIndex];
							weights += weight;
							sum += srcp[frameIndex][x] * weight;

							frameIndex--;
							v += 256;
						}
						else
						{
							break;
						}
					}
				}
			}

			frameIndex = _maxr + 1;

			if (frameIndex < toFrame) {
				int t1 = pfp[frameIndex][x];
				int diff = std::abs(c - t1);

				if (diff < thresh) {
					float weight = weightSaved[useDiff ? diff >> _shift : frameIndex];
					weights += weight;
					sum += srcp[frameIndex][x] * weight;

					frameIndex++;
					int v = 256;

					while (frameIndex < toFrame)
					{
						const int t2 = t1;
						t1 = pfp[frameIndex][x];
						diff = std::abs(c - t1);

						if (diff < thresh && std::abs(t1 - t2) < thresh)
						{
							weight = weightSaved[useDiff ? (diff >> _shift) + v : frameIndex];
							weights += weight;
							sum += srcp[frameIndex][x] * weight;

							frameIndex++;
							v += 256;
						}
						else
						{
							break;
						}
					}
				}
			}

			if (_fp)
				dstp[x] = static_cast<T>(srcp[_maxr][x] * (1.f - weights) + sum + 0.5f);
			else
				dstp[x] = static_cast<T>(sum / weights + 0.5f);
		}

		for (int i = 0; i < _diameter; i++)
		{
			srcp[i] += src_stride;
			pfp[i] += src_stride;
		}
		dstp += stride;
	}
}

template<bool useDiff>
void TTempSmooth::filterF(PVideoFrame src[15], PVideoFrame pf[15], PVideoFrame& dst, const int fromFrame, const int toFrame, const int plane, int src_stride, const TTempSmooth* const VS_RESTRICT, IScriptEnvironment* env) noexcept
{
	const int width = dst->GetRowSize(plane) / vi.ComponentSize();
	const int height = dst->GetHeight(plane);
	src_stride /= sizeof(float);
	const int stride = dst->GetPitch(plane) / sizeof(float);
	const float* srcp[15] = {}, * pfp[15] = {};
	for (int i = 0; i < _diameter; i++)
	{
		srcp[i] = reinterpret_cast<const float*>(src[i]->GetReadPtr(plane));
		pfp[i] = reinterpret_cast<const float*>(pf[i]->GetReadPtr(plane));
	}

	float* VS_RESTRICT dstp = reinterpret_cast<float*>(dst->GetWritePtr(plane));

	int l;
	if (plane == 1)
		l = 0;
	else if (plane == 2)
		l = 1;
	else
		l = 2;

	const float thresh = _threshF[l];

	const float* const weightSaved = _weight[l];

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const float c = pfp[_maxr][x];
			float weights = _cw;
			float sum = srcp[_maxr][x] * _cw;

			int frameIndex = _maxr - 1;

			if (frameIndex > fromFrame)
			{
				float t1 = pfp[frameIndex][x];
				float diff = min(std::abs(c - t1), 1.f);

				if (diff < thresh)
				{
					float weight = weightSaved[useDiff ? static_cast<int>(diff * 255.f) : frameIndex];
					weights += weight;
					sum += srcp[frameIndex][x] * weight;

					frameIndex--;
					int v = 256;

					while (frameIndex > fromFrame) {
						const float t2 = t1;
						t1 = pfp[frameIndex][x];
						diff = min(std::abs(c - t1), 1.f);

						if (diff < thresh && min(std::abs(t1 - t2), 1.f) < thresh)
						{
							weight = weightSaved[useDiff ? static_cast<int>(diff * 255.f) + v : frameIndex];
							weights += weight;
							sum += srcp[frameIndex][x] * weight;

							frameIndex--;
							v += 256;
						}
						else
						{
							break;
						}
					}
				}
			}

			frameIndex = _maxr + 1;

			if (frameIndex < toFrame)
			{
				float t1 = pfp[frameIndex][x];
				float diff = min(std::abs(c - t1), 1.f);

				if (diff < thresh) {
					float weight = weightSaved[useDiff ? static_cast<int>(diff * 255.f) : frameIndex];
					weights += weight;
					sum += srcp[frameIndex][x] * weight;

					frameIndex++;
					int v = 256;

					while (frameIndex < toFrame)
					{
						const float t2 = t1;
						t1 = pfp[frameIndex][x];
						diff = min(std::abs(c - t1), 1.f);

						if (diff < thresh && min(std::abs(t1 - t2), 1.f) < thresh)
						{
							weight = weightSaved[useDiff ? static_cast<int>(diff * 255.f) + v : frameIndex];
							weights += weight;
							sum += srcp[frameIndex][x] * weight;

							frameIndex++;
							v += 256;
						}
						else
						{
							break;
						}
					}
				}
			}

			if (_fp)
				dstp[x] = srcp[_maxr][x] * (1.f - weights) + sum;
			else
				dstp[x] = sum / weights;
		}

		for (int i = 0; i < _diameter; i++)
		{
			srcp[i] += src_stride;
			pfp[i] += src_stride;
		}
		dstp += stride;
	}
}

static void copy_plane(PVideoFrame& dst, PVideoFrame& src, int plane, IScriptEnvironment* env)
{
	const uint8_t* srcp = src->GetReadPtr(plane);
	int src_pitch = src->GetPitch(plane);
	int height = src->GetHeight(plane);
	int row_size = src->GetRowSize(plane);
	uint8_t* destp = dst->GetWritePtr(plane);
	int dst_pitch = dst->GetPitch(plane);
	env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

template<typename pixel_t>
static double get_sad_c(const uint8_t* c_plane8, const uint8_t* t_plane8, size_t height, size_t width, size_t c_pitch, size_t t_pitch) {
	const pixel_t* c_plane = reinterpret_cast<const pixel_t*>(c_plane8);
	const pixel_t* t_plane = reinterpret_cast<const pixel_t*>(t_plane8);
	c_pitch /= sizeof(pixel_t);
	t_pitch /= sizeof(pixel_t);
	typedef typename std::conditional < sizeof(pixel_t) == 4, double, int64_t>::type sum_t;
	sum_t accum = 0; // int32 holds sum of maximum 16 Mpixels for 8 bit, and 65536 pixels for uint16_t pixels

	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			accum += std::abs(t_plane[x] - c_plane[x]);
		}
		c_plane += c_pitch;
		t_plane += t_pitch;
	}

	return (double)accum;
}

static float ComparePlane(PVideoFrame& src, PVideoFrame& src1, const int pixelsize, const int bits_per_pixel, IScriptEnvironment* env)
{
	const uint8_t* srcp = src->GetReadPtr(PLANAR_Y);
	const uint8_t* srcp2 = src1->GetReadPtr(PLANAR_Y);
	int height = src->GetHeight(PLANAR_Y);
	int rowsize = src->GetRowSize(PLANAR_Y);
	int width = rowsize / pixelsize;
	int pitch = src->GetPitch(PLANAR_Y);
	int pitch2 = src1->GetPitch(PLANAR_Y);

	int total_pixels = width * height;
	bool sum_in_32bits;
	if (pixelsize == 4)
		sum_in_32bits = false;
	else // worst case check
		sum_in_32bits = ((int64_t)total_pixels * ((static_cast<int64_t>(1) << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();

	double sad = 0;
	// for c: width, for sse: rowsize
	if (pixelsize == 1)
		sad = get_sad_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
	else if (pixelsize == 2)
		sad = get_sad_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
	else // pixelsize==4
		sad = get_sad_c<float>(srcp, srcp2, height, width, pitch, pitch2);

	float f = static_cast<float>((sad / (static_cast<double>(height) * width)));

	if (bits_per_pixel != 32)
		f /= static_cast<float>((static_cast<int64_t>(1) << bits_per_pixel) - 1);

	return f;
}

TTempSmooth::TTempSmooth(PClip _child, int maxr, int ythresh, int uthresh, int vthresh, int ymdiff, int umdiff, int vmdiff, int strength, float scthresh, bool fp, bool y, bool u, bool v, PClip pfclip, IScriptEnvironment* env)
	: GenericVideoFilter(_child), _maxr(maxr), _ythresh(ythresh), _uthresh(uthresh), _vthresh(vthresh), _ymdiff(ymdiff), _umdiff(umdiff), _vmdiff(vmdiff), _scthresh(scthresh), _fp(fp), _y(y), _u(u), _v(v), _pfclip(pfclip)
{
	has_at_least_v8 = true;
	try { env->CheckVersion(8); }
	catch (const AvisynthError&) { has_at_least_v8 = false; }

	if (vi.IsRGB() || !vi.IsPlanar())
		env->ThrowError("vsTTempSmooth: clip must be Y/YUV(A) 8..32-bit planar format.");

	if (_maxr < 1 || _maxr > 7)
		env->ThrowError("vsTTempSmooth: maxr must be between 1..7.");

	if (_ythresh < 1 || _ythresh > 256)
		env->ThrowError("vsTTempSmooth: ythresh must be between 1..256.");

	if (_uthresh < 1 || _uthresh > 256)
		env->ThrowError("vsTTempSmooth: uthresh must be between 1..256.");

	if (_vthresh < 1 || _vthresh > 256)
		env->ThrowError("vsTTempSmooth: vthresh must be between 1..256.");

	if (_ymdiff < 0 || _ymdiff > 255)
		env->ThrowError("vsTTempSmooth: ymdiff must be between 0..255.");

	if (_umdiff < 0 || _umdiff > 255)
		env->ThrowError("vsTTempSmooth: umdiff must be between 0..255.");

	if (_vmdiff < 0 || _vmdiff > 255)
		env->ThrowError("vsTTempSmooth: vmdiff must be between 0..255.");

	if (strength < 1 || strength > 8)
		env->ThrowError("vsTTempSmooth: strength must be between 1..8.");

	if (_scthresh < 0.f || _scthresh > 100.f)
		env->ThrowError("vsTTempSmooth: scthresh must be between 0.0..100.0.");

	if (pfclip)
	{
		const VideoInfo& vi1 = pfclip->GetVideoInfo();
		if (!vi.IsSameColorspace(vi1) || vi.width != vi1.width || vi.height != vi1.height)
			env->ThrowError("vsTTempSmooth: pfclip must have the same dimension as the main clip and be the same format.");
		if (vi.num_frames != vi1.num_frames)
			env->ThrowError("vsTTempSmooth: pfclip's number of frames doesn't match.");
	}

	_cw = 0;
	*_mdiff = 0;
	*_thresh = 0;
	*_threshF = 0;
	*_weight = 0;
	*proccesplanes = 0;

	_shift = vi.BitsPerComponent() - 8;

	_diameter = _maxr * 2 + 1;

	planecount = min(vi.NumComponents(), 3);
	for (int i = 0; i < planecount; i++)
	{
		if (i == 0)
		{
			proccesplanes[i] = _y;
			_thresh[i] = _ythresh;
			_mdiff[i] = _ymdiff;
		}
		else if (i == 1)
		{
			proccesplanes[i] = _u;
			_thresh[i] = _uthresh;
			_mdiff[i] = _umdiff;
		}
		else if (i == 2)
		{
			proccesplanes[i] = _v;
			_thresh[i] = _vthresh;
			_mdiff[i] = _vmdiff;
		}

		if (proccesplanes[i])
		{
			if (_thresh[i] > _mdiff[i] + 1) {
				_weight[i] = new float[256 * static_cast<int64_t>(_maxr)];
				float dt[15] = {}, rt[256] = {}, sum = 0.f;

				for (int i = 0; i < strength && i <= _maxr; i++)
					dt[i] = 1.f;
				for (int i = strength; i <= _maxr; i++)
					dt[i] = 1.f / (i - strength + 2);

				const float step = 256.f / (_thresh[i] - min(_mdiff[i], _thresh[i] - 1));
				float base = 256.f;
				for (int j = 0; j < _thresh[i]; j++) {
					if (_mdiff[i] > j) {
						rt[j] = 256.f;
					}
					else {
						if (base > 0.f)
							rt[j] = base;
						else
							break;
						base -= step;
					}
				}

				sum += dt[0];
				for (int j = 1; j <= _maxr; j++) {
					sum += dt[j] * 2.f;
					for (int v = 0; v < 256; v++)
						_weight[i][256 * (j - 1) + v] = dt[j] * rt[v] / 256.f;
				}

				for (int j = 0; j < 256 * _maxr; j++)
					_weight[i][j] /= sum;

				_cw = dt[0] / sum;
			}
			else 
			{
				_weight[i] = new float[_diameter];
				float dt[15] = {}, sum = 0.f;

				for (int i = 0; i < strength && i <= _maxr; i++)
					dt[_maxr - i] = dt[_maxr + i] = 1.f;
				for (int i = strength; i <= _maxr; i++)
					dt[_maxr - i] = dt[_maxr + i] = 1.f / (i - strength + 2);

				for (int j = 0; j < _diameter; j++) {
					sum += dt[j];
					_weight[i][j] = dt[j];
				}

				for (int j = 0; j < _diameter; j++)
					_weight[i][j] /= sum;

				_cw = _weight[i][_maxr];
			}

			if (vi.BitsPerComponent() == 32)
				_threshF[i] = _thresh[i] / 256.f;
		}
	}
}

TTempSmooth::~TTempSmooth()
{
	for (int i = 0; i < planecount; i++)
		delete[] _weight[i];
}

PVideoFrame TTempSmooth::GetFrame(int n, IScriptEnvironment* env) {
	PVideoFrame src[15] = {};
	PVideoFrame pf[15] = {};
	PVideoFrame srcc = child->GetFrame(n, env);
	PVideoFrame srcp = child->GetFrame(n - 1, env);
	PVideoFrame srcn = child->GetFrame(n + 1, env);

	for (int i = n - _maxr; i <= n + _maxr; i++)
	{
		const int frameNumber = min(max(i, 0), vi.num_frames - 1);

		src[i - n + _maxr] = child->GetFrame(frameNumber, env);

		if (_pfclip)
			pf[i - n + _maxr] = _pfclip->GetFrame(frameNumber, env);
	}

	PVideoFrame dst;
	if (has_at_least_v8)
		dst = env->NewVideoFrameP(vi, &srcc);
	else
		dst = env->NewVideoFrame(vi);

	int fromFrame = -1, toFrame = _diameter;
	const int pixelsize = vi.ComponentSize();
	const int bits_per_pixel = vi.BitsPerComponent();

	if (_scthresh)
	{
		for (int i = _maxr; i > 0; i--)
		{
			if (ComparePlane(srcc, srcp, pixelsize, bits_per_pixel, env) > _scthresh / 100.f)
				fromFrame = i;
			break;
		}

		for (int i = _maxr; i < _diameter - 1; i++)
		{
			if (ComparePlane(srcc, srcn, pixelsize, bits_per_pixel, env) > _scthresh / 100.f)
				toFrame = i;
			break;
		}
	}

	int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
	const int* current_planes = planes_y;
	planecount = min(vi.NumComponents(), 3);
	for (int i = 0; i < planecount; i++)
	{
		const int plane = current_planes[i];

		int src_stride = srcc->GetPitch(plane);

		if (proccesplanes[i])
		{
			if (_thresh[i] > _mdiff[i] + 1)
			{
				if (vi.ComponentSize() == 1)
					TTempSmooth::filterI<uint8_t, true>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
				else if (vi.ComponentSize() == 2)
					TTempSmooth::filterI<uint16_t, true>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
				else
					TTempSmooth::filterF<true>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
			}
			else
			{
				if (vi.ComponentSize() == 1)
					TTempSmooth::filterI<uint8_t, false>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
				else if (vi.ComponentSize() == 2)
					TTempSmooth::filterI<uint16_t, false>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
				else
					TTempSmooth::filterF<false>(src, _pfclip ? pf : src, dst, fromFrame, toFrame, plane, src_stride, 0, env);
			}
		}
		else
			copy_plane(dst, srcc, plane, env);
	}

	return dst;
}

AVSValue __cdecl Create_TTempSmooth(AVSValue args, void* user_data, IScriptEnvironment* env)
{
	return new TTempSmooth(
		args[0].AsClip(),
		args[1].AsInt(3),
		args[2].AsInt(4),
		args[3].AsInt(5),
		args[4].AsInt(5),
		args[5].AsInt(2),
		args[6].AsInt(3),
		args[7].AsInt(3),
		args[8].AsInt(2),
		args[9].AsFloatf(12),
		args[10].AsBool(true),
		args[11].AsBool(true),
		args[12].AsBool(true),
		args[13].AsBool(true),
		args[14].AsClip(),
		env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;

	env->AddFunction("vsTTempSmooth", "c[maxr]i[ythresh]i[uthresh]i[vthresh]i[ymdiff]i[umdiff]i[vmdiff]i[strength]i[scthresh]f[fp]b[y]b[u]b[v]b[pfclip]c", Create_TTempSmooth, 0);
	return "vsTTempSmooth";
}

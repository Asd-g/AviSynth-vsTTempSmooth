## Description

TTempSmooth is a motion adaptive (it only works on stationary parts of the picture), temporal smoothing filter.

This is [a port of the VapourSynth plugin TTempSmooth](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TTempSmooth).

### Requirements:

- AviSynth 2.60 / AviSynth+ 3.4 or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
vsTTempSmooth(clip, int "ythresh", int "uthresh", int "vthresh", int "ymdiff", bool "umdiff" , bool "vmdiff", int "strength", float "scthresh", bool "fp", int "y", int "u", int "v", clip "pfclip", int "opt", int "pmode", int "ythupd", int "uthupd", int "vthupd", int "ypnew", int "upnew", int "vpnew", int "threads")
```

### Parameters:

- clip<br>
    A clip to process. It must be Y/YUV(A) 8..32-bit format.

- maxr<br>
    This sets the maximum temporal radius.<br>
    By the way it works TTempSmooth automatically varies the radius used...<br>
    This sets the maximum boundary.<br>
    `pmode=0` (Inverse Distance Weighting) or `pmode=2` (Gaussian/Binomial Weighting) - must be between 1 and 7.<br>
    `pmode=1` (Best Sample Selection) - must be between 1 and 128.<br>
    At 1 TTempSmooth will be (at max) including pixels from 1 frame away in the average (3 frames total will be considered counting the current frame).<br>
    At 7 it would be including pixels from up to 7 frames away (15 frames total will be considered).<br>
    With the way it checks motion there isn't much danger in setting this high (within the mode's limits), it's basically a quality vs. speed option. Lower settings are faster while larger values tend to create a more stable image.<br>
    Default: 3.

- ythresh<br>
    Luma threshold. Its application depends on `pmode`:
    - For `pmode=0` and `pmode=2`: It's a threshold for differences of pixels between frames. TTempSmooth checks 2 frame distance as well as single frame, so these can usually be set slightly higher than with most other temporal smoothers and still avoid artifacts. Also important is the fact that as long as `ymdiff` is less than `ythresh`, pixels with larger differences from the original will have less weight in the average.
    - For `pmode=1`: After the "best" sample is selected (from temporal neighbors or IIR memory), this threshold is used to compare the selected "best" sample with the original pixel in the current frame. If the difference is less than `ythresh`, the "best" sample is output; otherwise, the original pixel from the current frame is output.
    Must be between 1 and 256.<br>
    Default: 4.

- uthresh, vthresh<br>
    Same as ythresh but for the chroma planes (u, v).<br>
    Must be between 1 and 256.<br>
    Default: uthresh = 5; vthresh =5.

- ymdiff (only for `pmode=0` and `pmode=2`)<br>
    Any pixels with differences less than or equal to mdiff will be blurred at maximum.<br>
    Usually, the larger the difference to the center pixel the smaller the weight in the average.<br>
    mdiff makes TTempSmooth treat pixels that have a difference of less than or equal to mdiff as though they have a difference of 0. In other words, it shifts the zero difference point outwards.<br>
    Set mdiff to a value equal to or greater than thresh-1 to completely disable inverse pixel difference weighting.<br>
    Applied only to the luma plane.<br>
    Must be between 0 and 255.<br>
    Default: 2.

- umdiff, vmdiff (only for `pmode=0` and `pmode=2`)<br>
    Same as ymdiff but for the chroma planes (u, v).<br>
    Must be between 0 and 255.<br>
    Default: umdiff = 3; vmdiff = 3.

- strength (only for `pmode=0` and `pmode=2`)<br>
    Determines the shape of the distance-based weighting component.<br>
    - For `pmode=0`: TTempSmooth uses inverse distance weighting when deciding how much weight to give to each pixel value.<br>
    The strength option lets you shift the drop off point away from the center to give a stronger smoothing effect and add weight to the outer pixels.<br>
    It does for the spatial weights what mdiff does for the difference weights.

    - 1 = 0.13 0.14 0.16 0.20 0.25 0.33 0.50 1.00 0.50 0.33 0.25 0.20 0.16 0.14 0.13
    - 2 = 0.14 0.16 0.20 0.25 0.33 0.50 1.00 1.00 1.00 0.50 0.33 0.25 0.20 0.16 0.14
    - 3 = 0.16 0.20 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25 0.20 0.16
    - 4 = 0.20 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25 0.20
    - 5 = 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25
    - 6 = 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33
    - 7 = 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50
    - 8 = 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

    The values shown are for maxr=7, when using smaller radius values the weights outside of the range are simply dropped. Thus, setting strength to a value of maxr+1 or higher will give you equal spatial weighting of all pixels in the kernel.<br>
    - For `pmode=2`:
    - `strength = 1`: Pure binomial weighting is applied across all `2*maxr + 1` frames. The weights are derived from Pascal's triangle for `k = 2*maxr`. This gives a classic bell-curve shape.
    - `strength > 1`: A plateau is formed. Frames at distance `d < strength` (i.e., the center frame and `strength-1` frames on each side) share a common peak weight. The falloff from the edges of this plateau then follows a binomial curve shape. The peak weight of the plateau is determined by the peak of the binomial curve used for the falloff.
    Must be between 1 and 8.<br>
    Default: 2.

- scthresh<br>
    The standard scenechange threshold as a percentage of maximum possible change of the luma plane.<br>
    A good range of values is between 8 and 15.<br>
    Set to 0 to disable scenechange detection.<br>
    Must be between 0.0 and 100.0.<br>
    Default: 12.0.

- fp (only for `pmode=0` and `pmode=2`)<br>
    When true will add any weight not given to the outer pixels back onto the center pixel when computing the final value and it's much better for reducing artifacts in motion areas and usually produces overall better results. When false will just do a normal weighted average.<br>
    Default: True.

- y, u, v<br>
    Planes to process.<br>
    1: Return garbage.<br>
    2: Copy plane.<br>
    3: Process plane.<br>
    Default: y = 3, u = 3, v = 3.

- pfclip<br>
    This allows you to specify a separate clip for TTempSmooth to use when calculating pixel differences.<br>
    This applies to checking the motion thresholds, calculating inverse difference weights, and detecting scenechanges.<br>
    Basically, the pfclip will be used to determine the weights in the average but the weights will be applied to the original input clip's pixel values.

- opt<br>
    Sets which cpu optimizations to use.<br>
    -1: Auto-detect.<br>
    0: Use C++ code.<br>
    1: Use SSE2 code.<br>
    2: Use AVX2 code.<br>
    3: Use AVX-512 code.<br>
    Default: -1.

- pmode<br>
    Which mode to use.<br>
    0: Checking the motion thresholds and calculating inverse difference weights.<br>
    1: It tries to select "the best" sample from the total pool of input samples in the temporal axis for current position in frame using an algorithm of selecting "most equal/probable" value. More info [here](https://github.com/Asd-g/AviSynth-vsTTempSmooth/pull/8#issuecomment-1616112102).<br>
    `pmode=1` with thresholds = 255 is more or less equal to the MedianBlurTemporal AVS plugin. It is slower with `maxr=1` and possibly with `maxr=2`, and significantly faster with `maxr > 2` Also it provides additional functions of possible threshold to limit max difference with source sample and some optional temporal IIR filtering. More info [here](https://github.com/Asd-g/AviSynth-vsTTempSmooth/issues/7#issuecomment-2083117936).<br>
    2: Uses binomial coefficients (approximating a Gaussian curve) for the distance-based component of weights.<br>
    Default: 0.

- ythupd, uthupd, vthupd (only for `pmode=1`)<br>
    IIR memory update threshold respectively for luma and chroma planes.<br>
    If y/u/vthupd > 0 - enable IIR-type filtering. It remembers previous "best" sample output and compares current "best" with memory value. If the difference below thupd - the memory sample used for output. More info [here](https://github.com/Asd-g/AviSynth-vsTTempSmooth/pull/8#issuecomment-1616112102).<br>
    Must be greater than 0.<br>
    Default: 0.

- ypnew, upnew, vpnew (only for `pmode=1`)<br>
    Penalty for the new 'best' sample update in the IIR-memory respectively for luma and chroma planes.<br>
    It is added to the internal metric of the current "best" sample in the algorithm of replacing the current stored value with new "best".<br>
    Allow to decrease the number of memory replacement operations and create more stable temporal output. Close to idea of penalty to update current best motion vector with new candidate with a bit better metric in MAnalyse from mvtools.
    Must be greater than 0.<br>
    Default: 0.

- threads (only for `pmode=1`)<br>
    How many logical processors are used.<br>
    0: Maximum logical processors are used.<br>
    Must be between 0 and maximum logical processors.<br>
    Default: 0.

### Building:

```
Requirements:
- Git
- C++17 compiler
- CMake >= 3.16
- Ninja
```

```
git clone https://github.com/Asd-g/AviSynth-vsTTempSmooth
cd AviSynth-vsTTempSmooth
cmake -B build -G Ninja
ninja -C build
```

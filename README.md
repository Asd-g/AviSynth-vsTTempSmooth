# Description

TTempSmooth is a motion adaptive (it only works on stationary parts of the picture), temporal smoothing filter.

This is [a port of the VapourSynth plugin TTempSmooth](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TTempSmooth).

# Usage

```
vsTTempSmooth(clip, int "ythresh", int "uthresh", int "vthresh", int "ymdiff", bool "umdiff" , bool "vmdiff", int "strength", float "scthresh", bool "fp", bool "y", bool "u", bool "v", clip "pfclip")
```

## Parameters:

- clip\
    A clip to process. It must be Y/YUV(A) 8..32-bit format.
    
- maxr\
    This sets the maximum temporal radius. By the way it works TTempSmooth automatically varies the radius used... This sets the maximum boundary.\
    Must be between 1 and 7. At 1 TTempSmooth will be (at max) including pixels from 1 frame away in the average (3 frames total will be considered counting the current frame). At 7 it would be including pixels from up to 7 frames away (15 frames total will be considered). With the way it checks motion there isn't much danger in setting this high, it's basically a quality vs. speed option. Lower settings are faster while larger values tend to create a more stable image.\
    Default: 3.
    
- ythresh\
    Luma threshold for differences of pixels between frames. TTempSmooth checks 2 frame distance as well as single frame, so these can usually be set slightly higher than with most other temporal smoothers and still avoid artifacts.\
    Must be between 1 and 256.\
    Also important is the fact that as long as mdiff is less than the threshold value then pixels with larger differences from the original will have less weight in the average. Thus, even with rather large thresholds pixels just under the threshold wont have much weight, helping to reduce artifacts.\
    Default: 4.
    
- uthresh, vthresh\
    Same as ythresh but for the chroma planes (u, v).\
    Must be between 1 and 256.\
    Default: uthresh = 5; vthresh =5.
    
- ymdiff\
    Any pixels with differences less than or equal to mdiff will be blurred at maximum. Usually, the larger the difference to the center pixel the smaller the weight in the average. mdiff makes TTempSmooth treat pixels that have a difference of less than or equal to mdiff as though they have a difference of 0. In other words, it shifts the zero difference point outwards. Set mdiff to a value equal to or greater than thresh-1 to completely disable inverse pixel difference weighting.\
    Applied only to the luma plane.\
    Must be between 0 and 255.\
    Default: 2.
    
- umdiff, vmdiff\
    Same as ymdiff but for the chroma planes (u, v).\
    Must be between 0 and 255.\
    Default: umdiff = 3; vmdiff = 3.

- strength\
    TTempSmooth uses inverse distance weighting when deciding how much weight to give to each pixel value. The strength option lets you shift the drop off point away from the center to give a stronger smoothing effect and add weight to the outer pixels. It does for the spatial weights what mdiff does for the difference weights.

    - 1 = 0.13 0.14 0.16 0.20 0.25 0.33 0.50 1.00 0.50 0.33 0.25 0.20 0.16 0.14 0.13
    - 2 = 0.14 0.16 0.20 0.25 0.33 0.50 1.00 1.00 1.00 0.50 0.33 0.25 0.20 0.16 0.14
    - 3 = 0.16 0.20 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25 0.20 0.16
    - 4 = 0.20 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25 0.20
    - 5 = 0.25 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33 0.25
    - 6 = 0.33 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50 0.33
    - 7 = 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.50
    - 8 = 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

    The values shown are for maxr=7, when using smaller radius values the weights outside of the range are simply dropped. Thus, setting strength to a value of maxr+1 or higher will give you equal spatial weighting of all pixels in the kernel.\
    Must be between 1 and 8.\
    Default: 2.
    
- scthresh\
    The standard scenechange threshold as a percentage of maximum possible change of the luma plane. A good range of values is between 8 and 15.\
    Set to 0 to disable scenechange detection.\
    Must be between 0.0 and 100.0.\
    Default: 12.0.
    
- fp\
    When true will add any weight not given to the outer pixels back onto the center pixel when computing the final value and it's much better for reducing artifacts in motion areas and usually produces overall better results. When false will just do a normal weighted average.\
    Default: True.
    
- y, u, v\
    Planes to process.\
    Default: y = true; u = true; v = true.
    
- pfclip\
    This allows you to specify a separate clip for TTempSmooth to use when calculating pixel differences. This applies to checking the motion thresholds, calculating inverse difference weights, and detecting scenechanges. Basically, the pfclip will be used to determine the weights in the average but the weights will be applied to the original input clip's pixel values.

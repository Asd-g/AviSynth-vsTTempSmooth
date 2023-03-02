##### 1.2.2:
    Fixed used frames for scene change detection.
    Fixed memory allocation for inverse pixel difference weighting.
    Fixed inverse pixel difference weighting to work properly for every processed plane.

##### 1.2.1:
    Fixed processing with float clips. (regression from `1.2.0`)
    Added parameter `opt`.
    Added SSE2, AVX2, AVX-512 code.
    Fixed earlier exit of the scene change detection.
    Fixed memory misalignment.

##### 1.2.0:
    Fixed crash when releasing memory.
    Changed the type of parameters `y`, `u`, `v` to int.

##### 1.1.3:
    Throw error for non-planar formats.

##### 1.1.2:
    Fixed memory leak.

##### 1.1.1:
    Fixed memory misalignment for AviSynth 2.6.

##### 1.1.0:
    Added scthresh paramter.

##### 1.0.0:
    Port of the VapourSynth plugin TTempSmooth r3.1.

#ifndef SYCL_SUBGROUP_SIZE_H
#define SYCL_SUBGROUP_SIZE_H

#ifdef __SYCL_DEVICE_ONLY__

#    if defined(__SYCL_TARGET_INTEL_GPU_BDW__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_SKL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_KBL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_CFL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_APL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_GLK__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_WHL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_AML__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_CML__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_ICLLP__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_TGLLP__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_RKL__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_ADL_S__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_RPL_S__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_ADL_P__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_ADL_N__) || /* ... */                                                         \
        defined(__SYCL_TARGET_INTEL_GPU_DG1__) || /* ... */                                                           \
        defined(__SYCL_TARGET_INTEL_GPU_ACM_G10__) || /* ... */                                                       \
        defined(__SYCL_TARGET_INTEL_GPU_ACM_G11__) || /* ... */                                                       \
        defined(__SYCL_TARGET_INTEL_GPU_ACM_G12__) /* ... */

#        define SYCL_SUBGROUP_SIZE (8 | 16 | 32)

#    elif defined(__SYCL_TARGET_INTEL_GPU_PVC__) /* ... */

#        define SYCL_SUBGROUP_SIZE (16 | 32)

#    elif defined(__SYCL_TARGET_INTEL_X86_64__) /* ... */

#        define SYCL_SUBGROUP_SIZE (4 | 8 | 16 | 32 | 64)

#    elif defined(__SYCL_TARGET_NVIDIA_GPU_SM_50__) || /* ... */                                                      \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_52__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_53__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_60__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_61__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_62__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_70__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_72__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_75__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_80__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_86__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_87__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_89__) || /* ... */                                                        \
        defined(__SYCL_TARGET_NVIDIA_GPU_SM_90__)

#        define SYCL_SUBGROUP_SIZE 32

#    elif defined(__SYCL_TARGET_AMD_GPU_GFX700__) || /* ... */                                                        \
        defined(__SYCL_TARGET_AMD_GPU_GFX701__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX702__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX801__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX802__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX803__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX805__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX810__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX900__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX902__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX904__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX906__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX908__) || /* ... */                                                          \
        defined(__SYCL_TARGET_AMD_GPU_GFX90A__) /* ... */

#        define SYCL_SUBGROUP_SIZE 64

#    elif defined(__SYCL_TARGET_AMD_GPU_GFX1010__) || /* ... */                                                       \
        defined(__SYCL_TARGET_AMD_GPU_GFX1011__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1012__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1013__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1030__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1031__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1032__) || /* ... */                                                         \
        defined(__SYCL_TARGET_AMD_GPU_GFX1034__) /* ... */

#        define SYCL_SUBGROUP_SIZE 32

#    else // __SYCL_TARGET_*

#        define SYCL_SUBGROUP_SIZE 0

#    endif // __SYCL_TARGET_*

#    else

#        define SYCL_SUBGROUP_SIZE 0

#endif // __SYCL_DEVICE_ONLY__

#endif // SYCL_SUBGROUP_SIZE_H

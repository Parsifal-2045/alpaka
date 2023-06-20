/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

#    include <alpaka/acc/AccGenericSycl.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevGpuSyclNvidia.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/kernel/TaskKernelGpuSyclNvidia.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/PltfGpuSyclNvidia.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <string>
#    include <utility>

namespace alpaka
{
    //! The Nvidia GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Nvidia GPU target device.
    template<typename TDim, typename TIdx>
    class AccGpuSyclNvidia final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccGpuSyclNvidia<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The Nvidia GPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccGpuSyclNvidia<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccGpuSyclNvidia<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Nvidia GPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccGpuSyclNvidia<TDim, TIdx>>
    {
        using type = DevGpuSyclNvidia;
    };

    //! The Nvidia GPU SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccGpuSyclNvidia<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return TaskKernelGpuSyclNvidia<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Nvidia GPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<AccGpuSyclNvidia<TDim, TIdx>>
    {
        using type = PltfGpuSyclNvidia;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccGpuSyclNvidia<TDim, TIdx>>
    {
        using type = alpaka::TagGpuSyclNvidia;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagGpuSyclNvidia, TDim, TIdx>
    {
        using type = alpaka::AccGpuSyclNvidia<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif

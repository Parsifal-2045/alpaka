/* Copyright 2023 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>

#    include <string>

namespace alpaka
{
    namespace detail
    {
        // Prevent clang from annoying us with warnings about emitting too many vtables. These are discarded by the
        // linker anyway.
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
#    endif
        struct NvidiaGpuSelector : sycl::device_selector
        {
            auto operator()(sycl::device const& dev) const -> int override
            {
                auto const vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_nvidia_gpu = (vendor.find("NVIDIA") != std::string::npos) && dev.is_gpu();

                return is_nvidia_gpu ? 1 : -1;
            }
        };
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
    } // namespace detail

    //! The SYCL device manager.
    using PltfGpuSyclNvidia = PltfGenericSycl<detail::NvidiaGpuSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PltfGpuSyclNvidia>
    {
        using type = DevGenericSycl<PltfGpuSyclNvidia>; // = DevGpuSyclNvidia
    };
} // namespace alpaka::trait

#endif

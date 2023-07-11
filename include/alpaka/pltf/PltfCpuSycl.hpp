/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/pltf/PltfGenericSycl.hpp"

#include <string>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <CL/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        struct SyclCpuSelector
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_cpu() ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PltfCpuSycl = PltfGenericSycl<detail::SyclCpuSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PltfCpuSycl>
    {
        using type = DevGenericSycl<PltfCpuSycl>; // = DevCpuSycl
    };
} // namespace alpaka::trait

#endif

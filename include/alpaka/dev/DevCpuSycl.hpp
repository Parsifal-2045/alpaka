/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/pltf/PltfCpuSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    using DevCpuSycl = DevGenericSycl<PltfCpuSycl>;
} // namespace alpaka

#endif

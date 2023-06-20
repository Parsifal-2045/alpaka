/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <alpaka/dev/DevCpuSycl.hpp>
#    include <alpaka/event/EventGenericSycl.hpp>

namespace alpaka
{
    using EventCpuSycl = EventGenericSycl<DevCpuSycl>;
} // namespace alpaka

#endif

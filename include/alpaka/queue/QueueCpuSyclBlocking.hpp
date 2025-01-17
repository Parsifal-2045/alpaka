/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <alpaka/dev/DevCpuSycl.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>

namespace alpaka
{
    using QueueCpuSyclBlocking = QueueGenericSyclBlocking<DevCpuSycl>;
}

#endif

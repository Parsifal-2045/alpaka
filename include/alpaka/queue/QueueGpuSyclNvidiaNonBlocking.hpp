/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

#    include <alpaka/dev/DevGpuSyclNvidia.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>

namespace alpaka
{
    using QueueGpuSyclNvidiaNonBlocking = QueueGenericSyclNonBlocking<DevGpuSyclNvidia>;
}

#endif

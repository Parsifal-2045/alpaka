/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

namespace alpaka
{
    //! The HIP RT non-blocking queue.
    using QueueHipRtNonBlocking = QueueUniformCudaHipRtNonBlocking<ApiHipRt>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED

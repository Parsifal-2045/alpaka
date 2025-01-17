/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <alpaka/kernel/TaskKernelGenericSycl.hpp>

namespace alpaka
{
    template<typename TDim, typename TIdx>
    class AccFpgaSyclIntel;

    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelFpgaSyclIntel
        = TaskKernelGenericSycl<AccFpgaSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;
} // namespace alpaka

#endif

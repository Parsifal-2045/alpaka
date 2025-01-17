/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>

#    include <string>

namespace alpaka
{
    namespace detail
    {
        struct XilinxFpgaSelector
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_xilinx_fpga = dev.is_accelerator() && (vendor.find("Xilinx") != std::string::npos);

                return is_xilinx_fpga ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PltfFpgaSyclXilinx = PltfGenericSycl<detail::XilinxFpgaSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PltfFpgaSyclXilinx>
    {
        using type = DevGenericSycl<PltfFpgaSyclXilinx>; // = DevFpgaSyclXilinx
    };
} // namespace alpaka::trait

#endif

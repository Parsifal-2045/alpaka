/* Copyright 2023 Jan Stephan, Antonio Di Pilato, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

// Base classes.
#    include <alpaka/atomic/AtomicGenericSycl.hpp>
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStGenericSycl.hpp>
#    include <alpaka/block/sync/BlockSyncGenericSycl.hpp>
#    include <alpaka/idx/bt/IdxBtGenericSycl.hpp>
#    include <alpaka/idx/gb/IdxGbGenericSycl.hpp>
#    include <alpaka/intrinsic/IntrinsicGenericSycl.hpp>
#    include <alpaka/math/MathGenericSycl.hpp>
#    include <alpaka/mem/fence/MemFenceGenericSycl.hpp>
#    include <alpaka/rand/RandGenericSycl.hpp>
#    include <alpaka/warp/WarpGenericSycl.hpp>
#    include <alpaka/workdiv/WorkDivGenericSycl.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

// Implementation details.
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Sycl.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>
#    include <string>
#    include <type_traits>

namespace alpaka
{
    //! The SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL devices.
    template<typename TDim, typename TIdx>
    class AccGenericSycl
        : public WorkDivGenericSycl<TDim, TIdx>
        , public gb::IdxGbGenericSycl<TDim, TIdx>
        , public bt::IdxBtGenericSycl<TDim, TIdx>
        , public AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>
        , public math::MathGenericSycl
        , public BlockSharedMemDynGenericSycl
        , public BlockSharedMemStGenericSycl
        , public BlockSyncGenericSycl<TDim>
        , public IntrinsicGenericSycl
        , public MemFenceGenericSycl
        , public rand::RandGenericSycl<TDim>
        , public warp::WarpGenericSycl<TDim>
    {
    public:
        AccGenericSycl(AccGenericSycl const&) = delete;
        AccGenericSycl(AccGenericSycl&&) = delete;
        auto operator=(AccGenericSycl const&) -> AccGenericSycl& = delete;
        auto operator=(AccGenericSycl&&) -> AccGenericSycl& = delete;

#    ifdef ALPAKA_SYCL_IOSTREAM_ENABLED
        AccGenericSycl(
            Vec<TDim, TIdx> const& threadElemExtent,
            sycl::nd_item<TDim::value> work_item,
            sycl::local_accessor<std::byte> dyn_shared_acc,
            sycl::local_accessor<std::byte> st_shared_acc,
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> global_fence_dummy,
            sycl::local_accessor<int> local_fence_dummy,
            sycl::stream output_stream)
            : WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item}
            , gb::IdxGbGenericSycl<TDim, TIdx>{work_item}
            , bt::IdxBtGenericSycl<TDim, TIdx>{work_item}
            , AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{}
            , math::MathGenericSycl{}
            , BlockSharedMemDynGenericSycl{dyn_shared_acc}
            , BlockSharedMemStGenericSycl{st_shared_acc}
            , BlockSyncGenericSycl<TDim>{work_item}
            , IntrinsicGenericSycl{}
            , MemFenceGenericSycl{global_fence_dummy, local_fence_dummy}
            , rand::RandGenericSycl<TDim>{work_item}
            , warp::WarpGenericSycl<TDim>{work_item}
            , cout{output_stream}
        {
        }

        sycl::stream cout;
#    else
        AccGenericSycl(
            Vec<TDim, TIdx> const& threadElemExtent,
            sycl::nd_item<TDim::value> work_item,
            sycl::local_accessor<std::byte> dyn_shared_acc,
            sycl::local_accessor<std::byte> st_shared_acc,
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> global_fence_dummy,
            sycl::local_accessor<int> local_fence_dummy)
            : WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item}
            , gb::IdxGbGenericSycl<TDim, TIdx>{work_item}
            , bt::IdxBtGenericSycl<TDim, TIdx>{work_item}
            , AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{}
            , math::MathGenericSycl{}
            , BlockSharedMemDynGenericSycl{dyn_shared_acc}
            , BlockSharedMemStGenericSycl{st_shared_acc}
            , BlockSyncGenericSycl<TDim>{work_item}
            , IntrinsicGenericSycl{}
            , MemFenceGenericSycl{global_fence_dummy, local_fence_dummy}
            , rand::RandGenericSycl<TDim>{work_item}
            , warp::WarpGenericSycl<TDim>{work_item}
        {
        }
#    endif
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL accelerator type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct AccType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TAcc<TDim, TIdx>;
    };

    //! The SYCL accelerator device properties get trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct GetAccDevProps<
        TAcc<TDim, TIdx>,
        std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        static auto getAccDevProps(typename DevType<TAcc<TDim, TIdx>>::type const& dev) -> AccDevProps<TDim, TIdx>
        {
            auto const device = dev.getNativeHandle().first;
            auto const max_threads_dim
                = device.template get_info<sycl::info::device::max_work_item_sizes<TDim::value>>();
            Vec<TDim, TIdx> max_threads_dim_vec{};
            for(int i = 0; i < static_cast<int>(TDim::value); i++)
                max_threads_dim_vec[i] = alpaka::core::clipCast<TIdx>(max_threads_dim[i]);
            return {// m_multiProcessorCount
                    alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_compute_units>()),
                    // m_gridBlockExtentMax
                    getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                        // WARNING: There is no SYCL way to determine these values
                        std::numeric_limits<TIdx>::max(),
                        std::numeric_limits<TIdx>::max(),
                        std::numeric_limits<TIdx>::max())),
                    // m_gridBlockCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_blockThreadExtentMax
                    max_threads_dim_vec,
                    // m_blockThreadCountMax
                    alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    device.template get_info<sycl::info::device::local_mem_size>()};
        }
    };

    //! The SYCL accelerator dimension getter trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct DimType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator idx type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct IdxType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif

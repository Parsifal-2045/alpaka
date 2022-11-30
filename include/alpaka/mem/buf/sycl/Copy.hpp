/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Debug.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/elem/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/buf/sycl/Common.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>

#    include <CL/sycl.hpp>

#    include <memory>
#    include <type_traits>

namespace alpaka::detail
{
    //!  The Sycl device memory copy task base.
    template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySyclBase
    {
        static_assert(
            std::is_same_v<std::remove_const<alpaka::Elem<TViewSrc>>, std::remove_const<alpaka::Elem<TViewDst>>>,
            "Source and destination should be of the same type");
            
        using ExtentSize = Idx<TExtent>;
        using DstSize = Idx<TViewDst>;
        using SrcSize = Idx<TViewSrc>;
        using Elem = alpaka::Elem<TViewSrc>;

        template<typename TViewFwd>
        TaskCopySyclBase(TViewFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
            : m_extent(getExtentVec(extent))
            , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            , m_dstExtent(getExtentVec(viewDst))
            , m_srcExtent(getExtentVec(viewSrc))
#    endif
            , m_dstPitchBytes(getPitchBytesVec(viewDst))
            , m_srcPitchBytes(getPitchBytesVec(viewSrc))
            , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
            , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
        {
            if constexpr(TDim::value > 0)
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT((castVec<SrcSize>(m_extent) <= m_srcExtent).foldrAll(std::logical_or<bool>()));
            }
        }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        ALPAKA_FN_HOST auto printDebug() const -> void
        {
            std::cout << __func__ << " e: " << m_extent << " ewb: " << this->m_extentWidthBytes
                      << " de: " << m_dstExtent << " dptr: " << reinterpret_cast<void*>(m_dstMemNative)
                      << " se: " << m_srcExtent << " sptr: " << reinterpret_cast<void const*>(m_srcMemNative)
                      << std::endl;
        }
#    endif

        Vec<TDim, ExtentSize> const m_extent;
        ExtentSize const m_extentWidthBytes;
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
        Vec<TDim, DstSize> const m_dstExtent;
        Vec<TDim, SrcSize> const m_srcExtent;
#    endif

        Vec<TDim, DstSize> const m_dstPitchBytes;
        Vec<TDim, SrcSize> const m_srcPitchBytes;
        std::uint8_t* const m_dstMemNative;
        std::uint8_t const* const m_srcMemNative;
    };

    //! The Sycl device ND memory copy task.
    template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl : public TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>
    {
        using DimMin1 = DimInt<TDim::value - 1u>;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::ExtentSize;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::DstSize;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::SrcSize;

        using TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::TaskCopySyclBase;

        template<typename TQueue>
        ALPAKA_FN_HOST auto enqueue(TQueue& queue) const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            this->printDebug();
#    endif
            // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one
            // iteration.
            Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
            // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
            Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_dstPitchBytes));
            Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_srcPitchBytes));

            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
            {
                meta::ndLoopIncIdx(
                    extentWithoutInnermost,
                    [&](Vec<DimMin1, ExtentSize> const& idx)
                    {
                        queue.getNativeHandle().memcpy(
                            reinterpret_cast<void*>(
                                this->m_dstMemNative
                                + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<DstSize>())),
                            reinterpret_cast<void const*>(
                                this->m_srcMemNative
                                + (castVec<SrcSize>(idx) * srcPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<SrcSize>())),
                            static_cast<std::size_t>(this->m_extentWidthBytes));
                    });
            }
        }
    };

    //! The SYCL device 1D memory copy task.
    template<typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent>
        : TaskCopySyclBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>
    {
        using TaskCopySyclBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>::TaskCopySyclBase;

        template<typename TQueue>
        ALPAKA_FN_HOST auto enqueue(TQueue& queue) const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            this->printDebug();
#    endif
            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
            {
                queue.getNativeHandle().memcpy(
                    reinterpret_cast<void*>(this->m_dstMemNative),
                    reinterpret_cast<void const*>(this->m_srcMemNative),
                    static_cast<std::size_t>(this->m_extentWidthBytes));
            }
        }
    };

    //! The scalar SYCL memory copy trait.
    template<typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent>
    {
        static_assert(
            std::is_same_v<std::remove_const<alpaka::Elem<TViewSrc>>, std::remove_const<alpaka::Elem<TViewDst>>>,
            "Source and destination should be of the same type");

        using Elem = alpaka::Elem<TViewSrc>;

        template<typename TViewDstFwd>
        ALPAKA_FN_HOST TaskCopySycl(
            TViewDstFwd&& viewDst,
            TViewSrc const& viewSrc,
            [[maybe_unused]] TExtent const& extent)
            : m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
            , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
        {
            // all zero-sized extents are equivalent
            ALPAKA_ASSERT(getExtentVec(extent).prod() == 1u);
            ALPAKA_ASSERT(getExtentVec(viewDst).prod() == 1u);
            ALPAKA_ASSERT(getExtentVec(viewSrc).prod() == 1u);
        }

        template<typename TQueue>
        auto enqueue(TQueue& queue) const -> void
        {
            queue.getNativeHandle().memcpy(m_dstMemNative, m_srcMemNative, sizeof(Elem));
        }

    private:
        void* m_dstMemNative;
        void const* m_srcMemNative;
    };
} // namespace alpaka::detail

// Trait specializations for CreateTaskMemcpy.
namespace alpaka::trait
{
    //! The SYCL host-to-device memory copy trait specialization.
    template<typename TPltf, typename TDim>
    struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltf>, DevCpu>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        ALPAKA_FN_HOST static auto createTaskMemcpy(
            TViewDstFwd&& viewDst,
            TViewSrc const& viewSrc,
            TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };

    //! The SYCL device-to-host memory copy trait specialization.
    template<typename TPltf, typename TDim>
    struct CreateTaskMemcpy<TDim, DevCpu, DevGenericSycl<TPltf>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        ALPAKA_FN_HOST static auto createTaskMemcpy(
            TViewDstFwd&& viewDst,
            TViewSrc const& viewSrc,
            TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };

    //! The SYCL device-to-device memory copy trait specialization.
    template<typename TPltfDst, typename TPltfSrc, typename TDim>
    struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltfDst>, DevGenericSycl<TPltfSrc>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        ALPAKA_FN_HOST static auto createTaskMemcpy(
            TViewDstFwd&& viewDst,
            TViewSrc const& viewSrc,
            TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };

    //! The SYCL non-blocking device queue scalar copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclNonBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
        }
    };

    //! The SYCL blocking device queue scalar copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
            queue.getNativeHandle().wait();
        }
    };

    //! The SYCL non-blocking device queue 1D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclNonBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
        }
    };

    //! The SYCL blocking device queue 1D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
            queue.getNativeHandle().wait();
        }
    };

    //! The SYCL non-blocking device queue 2D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclNonBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<2u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
        }
    };

    //! The SYCL blocking device queue 2D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<2u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
            queue.getNativeHandle().wait();
        }
    };

    //! The SYCL non-blocking device queue 3D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclNonBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<3u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
        }
    };

    //! The SYCL blocking device queue 3D copy enqueue trait specialization.
    template<typename TPltf, typename TExtent, typename TViewSrc, typename TViewDst>
    struct Enqueue<
        alpaka::QueueGenericSyclBlocking<TPltf>,
        alpaka::detail::TaskCopySycl<DimInt<3u>, TViewDst, TViewSrc, TExtent>>
    {
        ALPAKA_FN_HOST static auto enqueue(
            alpaka::QueueGenericSyclBlocking<TPltf>& queue,
            alpaka::detail::TaskCopySycl<DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            task.enqueue(queue);
            queue.getNativeHandle().wait();
        }
    };

} // namespace alpaka::trait

#endif

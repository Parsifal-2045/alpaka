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
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/buf/sycl/Common.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#    include <alpaka/queue/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>
#    include <cstdint>
#    include <memory>

namespace alpaka
{
    namespace detail
    {
        //!  The SYCL memory set task base.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetSyclBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            TaskSetSyclBase(TView& view, std::uint8_t const& byte, TExtent const& extent)
                : m_view(view)
                , m_byte(byte)
                , m_extent(extent)
            {
            }

        protected:
            TView& m_view;
            std::uint8_t const m_byte;
            TExtent const m_extent;
        };

        //! The SYCL memory set task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetSycl;

        //! The scalar SYCL memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetSycl<DimInt<0u>, TView, TExtent> : public TaskSetSyclBase<DimInt<0u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetSycl(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetSyclBase<DimInt<0u>, TView, TExtent>(std::forward<TViewFwd>(view), byte, extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                queue.getNativeHandle().memset(
                    getPtrNative(this->m_view),
                    static_cast<int>(this->m_byte),
                    sizeof(Elem<TView>));
            }
        };

        //! The 1D SYCL memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetSycl<DimInt<1u>, TView, TExtent> : public TaskSetSyclBase<DimInt<1u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetSycl(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetSyclBase<DimInt<1u>, TView, TExtent>(std::forward<TViewFwd>(view), byte, extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                using Idx = Idx<TExtent>;

                auto& view = this->m_view;
                auto const& extent = this->m_extent;

                auto const extentWidth = getWidth(extent);
                ALPAKA_ASSERT(extentWidth <= getWidth(view));

                if(extentWidth == 0)
                {
                    return;
                }

                // Initiate the memory set.
                auto const extentWidthBytes = extentWidth * static_cast<Idx>(sizeof(Elem<TView>));

                queue.getNativeHandle().memset(
                    getPtrNative(this->m_view),
                    static_cast<int>(this->m_byte),
                    static_cast<size_t>(extentWidthBytes));
            }
        };


    } // namespace detail

    namespace trait
    {
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemset<TDim, DevGenericSycl<TPltf>>
        {
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
                -> alpaka::detail::TaskSetSycl<TDim, TView, TExtent>
            {
                return alpaka::detail::TaskSetSycl<TDim, TView, TExtent>(view, byte, extent);
            }
        };

        //! The SYCL non-blocking device queue scalar set enqueue trait specialization.
        template<typename TView, typename TExtent, typename TPltf>
        struct Enqueue<
            alpaka::QueueGenericSyclNonBlocking<TPltf>,
            alpaka::detail::TaskSetSycl<DimInt<0u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
                alpaka::detail::TaskSetSycl<DimInt<0u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The SYCL blocking device queue scalar set enqueue trait specialization.
        template<typename TView, typename TExtent, typename TPltf>
        struct Enqueue<
            alpaka::QueueGenericSyclBlocking<TPltf>,
            alpaka::detail::TaskSetSycl<DimInt<0u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::QueueGenericSyclBlocking<TPltf>& queue,
                alpaka::detail::TaskSetSycl<DimInt<0u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
                queue.getNativeHandle().wait();
            }
        };

        //! The SYCL non-blocking device queue 1D set enqueue trait specialization.
        template<typename TView, typename TExtent, typename TPltf>
        struct Enqueue<
            alpaka::QueueGenericSyclNonBlocking<TPltf>,
            alpaka::detail::TaskSetSycl<DimInt<1u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::QueueGenericSyclNonBlocking<TPltf>& queue,
                alpaka::detail::TaskSetSycl<DimInt<1u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The SYCL blocking device queue 1D set enqueue trait specialization.
        template<typename TView, typename TExtent, typename TPltf>
        struct Enqueue<
            alpaka::QueueGenericSyclBlocking<TPltf>,
            alpaka::detail::TaskSetSycl<DimInt<1u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                alpaka::QueueGenericSyclBlocking<TPltf>& queue,
                alpaka::detail::TaskSetSycl<DimInt<1u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
                queue.getNativeHandle().wait();
            }
        };

    } // namespace trait

} // namespace alpaka
#endif
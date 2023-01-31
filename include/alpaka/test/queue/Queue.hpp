/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::test
{
    namespace trait
    {
        //! The default queue type trait for devices.
        template<typename TDev, typename TSfinae = void>
        struct DefaultQueueType;

        //! The default queue type trait specialization for the CPU device.
        template<>
        struct DefaultQueueType<DevCpu>
        {
#if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueCpuBlocking;
#else
            using type = QueueCpuNonBlocking;
#endif
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

        //! The default queue type trait specialization for the CUDA/HIP device.
        template<typename TApi>
        struct DefaultQueueType<DevUniformCudaHipRt<TApi>>
        {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueUniformCudaHipRtBlocking<TApi>;
#    else
            using type = QueueUniformCudaHipRtNonBlocking<TApi>;
#    endif
        };
#endif
    } // namespace trait

    //! The queue type that should be used for the given device.
    template<typename TDev>
    using DefaultQueue = typename trait::DefaultQueueType<TDev>::type;

    namespace trait
    {
        //! The blocking queue trait.
        template<typename TQueue, typename TSfinae = void>
        struct IsBlockingQueue;

        //! The blocking queue trait specialization for a blocking CPU queue.
        template<typename TDev>
        struct IsBlockingQueue<QueueGenericThreadsBlocking<TDev>>
        {
            static constexpr bool value = true;
        };

        //! The blocking queue trait specialization for a non-blocking CPU queue.
        template<typename TDev>
        struct IsBlockingQueue<QueueGenericThreadsNonBlocking<TDev>>
        {
            static constexpr bool value = false;
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

        //! The blocking queue trait specialization for a blocking CUDA/HIP RT queue.
        template<typename TApi>
        struct IsBlockingQueue<QueueUniformCudaHipRtBlocking<TApi>>
        {
            static constexpr bool value = true;
        };

        //! The blocking queue trait specialization for a non-blocking CUDA/HIP RT queue.
        template<typename TApi>
        struct IsBlockingQueue<QueueUniformCudaHipRtNonBlocking<TApi>>
        {
            static constexpr bool value = false;
        };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

        //! The default queue type trait specialization for the Omp5 device.
        template<>
        struct DefaultQueueType<DevOmp5>
        {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueOmp5Blocking;
#    else
            using type = QueueOmp5Blocking;
#    endif
        };
#elif defined ALPAKA_ACC_ANY_BT_OACC_ENABLED

        //! The default queue type trait specialization for the OMP4 device.
        template<>
        struct DefaultQueueType<DevOacc>
        {
            using type = QueueOaccBlocking;
        };
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
        //! The default queue type trait specialization for the Intel CPU device.
        template<>
        struct DefaultQueueType<alpaka::DevCpuSyclIntel>
        {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = alpaka::QueueCpuSyclIntelBlocking;
#            else
            using type = alpaka::QueueCpuSyclIntelNonBlocking;
#            endif
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueCpuSyclIntelBlocking>
        {
            static constexpr auto value = true;
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueCpuSyclIntelNonBlocking>
        {
            static constexpr auto value = false;
        };
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
        //! The default queue type trait specialization for the Intel SYCL device.
        template<>
        struct DefaultQueueType<alpaka::DevFpgaSyclIntel>
        {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = alpaka::QueueFpgaSyclIntelBlocking;
#            else
            using type = alpaka::QueueFpgaSyclIntelNonBlocking;
#            endif
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueFpgaSyclIntelBlocking>
        {
            static constexpr auto value = true;
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueFpgaSyclIntelNonBlocking>
        {
            static constexpr auto value = false;
        };
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
        //! The default queue type trait specialization for the Intel CPU device.
        template<>
        struct DefaultQueueType<alpaka::DevGpuSyclIntel>
        {
#            if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = alpaka::QueueGpuSyclIntelBlocking;
#            else
            using type = alpaka::QueueGpuSyclIntelNonBlocking;
#            endif
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueGpuSyclIntelBlocking>
        {
            static constexpr auto value = true;
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueGpuSyclIntelNonBlocking>
        {
            static constexpr auto value = false;
        };
#        endif
#    endif
#    ifdef ALPAKA_SYCL_BACKEND_XILINX
        //! The default queue type trait specialization for the Xilinx SYCL device.
        template<>
        struct DefaultQueueType<alpaka::DevFpgaSyclXilinx>
        {
#        if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = alpaka::QueueFpgaSyclXilinxBlocking;
#        else
            using type = alpaka::QueueFpgaSyclXilinxNonBlocking;
#        endif
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueFpgaSyclXilinxBlocking>
        {
            static constexpr auto value = true;
        };

        template<>
        struct IsBlockingQueue<alpaka::QueueFpgaSyclXilinxNonBlocking>
        {
            static constexpr auto value = false;
        };
#    endif
#endif
    } // namespace trait
    //! The queue type that should be used for the given accelerator.
    template<typename TQueue>
    using IsBlockingQueue = trait::IsBlockingQueue<TQueue>;

    //! A std::tuple holding tuples of devices and corresponding queue types.
    using TestQueues = std::tuple<
        std::tuple<DevCpu, QueueCpuBlocking>,
        std::tuple<DevCpu, QueueCpuNonBlocking>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        ,
        std::tuple<DevCudaRt, QueueCudaRtBlocking>,
        std::tuple<DevCudaRt, QueueCudaRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        ,
        std::tuple<DevHipRt, QueueHipRtBlocking>,
        std::tuple<DevHipRt, QueueHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        ,
        std::tuple<DevOmp5, QueueOmp5Blocking>,
        std::tuple<DevOmp5, QueueOmp5NonBlocking>
#elif defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)
        ,
        std::tuple<DevOacc, QueueOaccBlocking>,
        std::tuple<DevOacc, QueueOaccNonBlocking>
#endif
#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_BACKEND_ONEAPI
#        ifdef ALPAKA_SYCL_ONEAPI_CPU
        ,
        std::tuple<alpaka::DevCpuSyclIntel, alpaka::QueueCpuSyclIntelBlocking>,
        std::tuple<alpaka::DevCpuSyclIntel, alpaka::QueueCpuSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_FPGA
        ,
        std::tuple<alpaka::DevFpgaSyclIntel, alpaka::QueueFpgaSyclIntelBlocking>,
        std::tuple<alpaka::DevFpgaSyclIntel, alpaka::QueueFpgaSyclIntelNonBlocking>
#        endif
#        ifdef ALPAKA_SYCL_ONEAPI_GPU
        ,
        std::tuple<alpaka::DevGpuSyclIntel, alpaka::QueueGpuSyclIntelBlocking>,
        std::tuple<alpaka::DevGpuSyclIntel, alpaka::QueueGpuSyclIntelNonBlocking>
#        endif
#    endif
#    if defined(ALPAKA_SYCL_BACKEND_XILINX)
        ,
        std::tuple<alpaka::DevFpgaSyclXilinx, alpaka::QueueFpgaSyclXilinxBlocking>,
        std::tuple<alpaka::DevFpgaSyclXilinx, alpaka::QueueFpgaSyclXilinxNonBlocking>
#    endif
#endif
        >;
} // namespace alpaka::test

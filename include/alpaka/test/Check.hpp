/* Copyright 2023 Benjamin Worpitz, Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdio>

// TODO: SYCL doesn't have a way to detect if we're looking at device or host code. This needs a workaround so that
// SYCL and other back-ends are compatible.
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_IOSTREAM_ENABLED)
#    define ALPAKA_CHECK(success, expression)                                                                         \
        do                                                                                                            \
        {                                                                                                             \
            if(!(expression))                                                                                         \
            {                                                                                                         \
                acc.cout << "ALPAKA_CHECK failed because '!(" << #expression << ")'\n";                               \
                success = false;                                                                                      \
            }                                                                                                         \
        } while(0)
#else
#    define ALPAKA_CHECK(success, expression)                                                                         \
        do                                                                                                            \
        {                                                                                                             \
            if(!(expression))                                                                                         \
            {                                                                                                         \
                printf("ALPAKA_CHECK failed because '!(%s)'\n", #expression);                                         \
                success = false;                                                                                      \
            }                                                                                                         \
        } while(0)
#endif

/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Vec.hpp>  // alpaka::vec

//-----------------------------------------------------------------------------
//! The name space for the accelerator library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //#############################################################################
    //! The trait for getting the size of the block shared extern memory for a kernel.
    //#############################################################################
    template<typename TAccelereatedKernel>
    struct BlockSharedExternMemSizeBytes
    {
        //-----------------------------------------------------------------------------
        //! \tparam TArgs The kernel invocation argument types pack.
        //! \param v3uiBlockKernelsExtent The size of the blocks for which the block shared memory size should be calculated.
        //! \param ... The kernel invocation arguments for which the block shared memory size should be calculated.
        //! \return The size of the shared memory allocated for a block in bytes.
        //! The default version always returns zero.
        //-----------------------------------------------------------------------------
        template<typename... TArgs>
        static std::size_t getBlockSharedExternMemSizeBytes(vec<3u> const & /*v3uiBlockKernelsExtent*/, TArgs && ...)
        {
            return 0;
        }
    };
}

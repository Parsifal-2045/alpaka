/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <cmath>

namespace alpaka::math
{
    struct ConceptMathAbs
    {
    };

    struct ConceptMathAcos
    {
    };

    struct ConceptMathAsin
    {
    };

    struct ConceptMathAtan
    {
    };

    struct ConceptMathAtan2
    {
    };

    struct ConceptMathCbrt
    {
    };

    struct ConceptMathCeil
    {
    };

    struct ConceptMathCos
    {
    };

    struct ConceptMathErf
    {
    };

    struct ConceptMathExp
    {
    };

    struct ConceptMathFloor
    {
    };

    struct ConceptMathFmod
    {
    };

    struct ConceptMathIsfinite
    {
    };

    struct ConceptMathIsinf
    {
    };

    struct ConceptMathIsnan
    {
    };

    struct ConceptMathLog
    {
    };

    struct ConceptMathMax
    {
    };

    struct ConceptMathMin
    {
    };

    struct ConceptMathPow
    {
    };

    struct ConceptMathRemainder
    {
    };

    struct ConceptMathRound
    {
    };

    struct ConceptMathRsqrt
    {
    };

    struct ConceptMathSin
    {
    };

    struct ConceptMathSinCos
    {
    };

    struct ConceptMathSqrt
    {
    };

    struct ConceptMathTan
    {
    };

    struct ConceptMathTrunc
    {
    };

    //! The math traits.
    namespace traits
    {
        //! The abs trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Abs
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find abs(TArg) in the namespace of your type.
                using std::abs;
                return abs(arg);
            }
        };

        //! The acos trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Acos
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find acos(TArg) in the namespace of your type.
                using std::acos;
                return acos(arg);
            }
        };

        //! The asin trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Asin
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find asin(TArg) in the namespace of your type.
                using std::asin;
                return asin(arg);
            }
        };

        //! The atan trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Atan
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find atan(TArg) in the namespace of your type.
                using std::atan;
                return atan(arg);
            }
        };

        //! The atan2 trait.
        template<typename T, typename Ty, typename Tx, typename TSfinae = void>
        struct Atan2
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, Ty const& y, Tx const& x)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find atan2(Tx, Ty) in the namespace of your type.
                using std::atan2;
                return atan2(y, x);
            }
        };

        //! The cbrt trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Cbrt
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find cbrt(TArg) in the namespace of your type.
                using std::cbrt;
                return cbrt(arg);
            } //! The erf trait.
        };

        //! The ceil trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Ceil
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find ceil(TArg) in the namespace of your type.
                using std::ceil;
                return ceil(arg);
            }
        };

        //! The cos trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Cos
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find cos(TArg) in the namespace of your type.
                using std::cos;
                return cos(arg);
            }
        };

        template<typename T, typename TArg, typename TSfinae = void>
        struct Erf
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find erf(TArg) in the namespace of your type.
                using std::erf;
                return erf(arg);
            }
        };

        //! The exp trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Exp
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find exp(TArg) in the namespace of your type.
                using std::exp;
                return exp(arg);
            }
        };

        //! The floor trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Floor
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find floor(TArg) in the namespace of your type.
                using std::floor;
                return floor(arg);
            }
        };

        //! The fmod trait.
        template<typename T, typename Tx, typename Ty, typename TSfinae = void>
        struct Fmod
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find fmod(Tx, Ty) in the namespace of your type.
                using std::fmod;
                return fmod(x, y);
            }
        };

        //! The isfinite trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Isfinite
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isfinite(TArg) in the namespace of your type.
                using std::isfinite;
                return isfinite(arg);
            }
        };

        //! The isinf trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Isinf
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isinf(TArg) in the namespace of your type.
                using std::isinf;
                return isinf(arg);
            }
        };

        //! The isnan trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Isnan
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isnan(TArg) in the namespace of your type.
                using std::isnan;
                return isnan(arg);
            }
        };

        //! The log trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Log
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find log(TArg) in the namespace of your type.
                using std::log;
                return log(arg);
            }
        };

        //! The max trait.
        template<typename T, typename Tx, typename Ty, typename TSfinae = void>
        struct Max
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find max(Tx, Ty) in the namespace of your type.
                using std::max;
                return max(x, y);
            }
        };

        //! The min trait.
        template<typename T, typename Tx, typename Ty, typename TSfinae = void>
        struct Min
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find min(Tx, Ty) in the namespace of your type.
                using std::min;
                return min(x, y);
            }
        };

        //! The pow trait.
        template<typename T, typename TBase, typename TExp, typename TSfinae = void>
        struct Pow
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TBase const& base, TExp const& exp)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find pow(base, exp) in the namespace of your type.
                using std::pow;
                return pow(base, exp);
            }
        };

        //! The remainder trait.
        template<typename T, typename Tx, typename Ty, typename TSfinae = void>
        struct Remainder
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find remainder(Tx, Ty) in the namespace of your type.
                using std::remainder;
                return remainder(x, y);
            }
        };

        //! The round trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Round
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find round(TArg) in the namespace of your type.
                using std::round;
                return round(arg);
            }
        };

        //! The round trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Lround
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find lround(TArg) in the namespace of your type.
                using std::lround;
                return lround(arg);
            }
        };

        //! The round trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Llround
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find llround(TArg) in the namespace of your type.
                using std::llround;
                return llround(arg);
            }
        };

        namespace detail
        {
            //! Fallback implementation when no better ADL match was found
            template<typename TArg>
            ALPAKA_FN_HOST_ACC auto rsqrt(TArg const& arg)
            {
                // Still use ADL to try find sqrt(arg)
                using std::sqrt;
                return static_cast<TArg>(1) / sqrt(arg);
            }
        } // namespace detail

        //! The rsqrt trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Rsqrt
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find rsqrt(TArg) in the namespace of your type.
                using detail::rsqrt;
                return rsqrt(arg);
            }
        };

        //! The sin trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Sin
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sin(TArg) in the namespace of your type.
                using std::sin;
                return sin(arg);
            }
        };

        namespace detail
        {
            //! Fallback implementation when no better ADL match was found
            template<typename TArg>
            ALPAKA_FN_HOST_ACC auto sincos(TArg const& arg, TArg& result_sin, TArg& result_cos)
            {
                // Still use ADL to try find sin(arg) and cos(arg)
                using std::sin;
                result_sin = sin(arg);
                using std::cos;
                result_cos = cos(arg);
            }
        } // namespace detail

        //! The sincos trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct SinCos
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg, TArg& result_sin, TArg& result_cos)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sincos(TArg, TArg&, TArg&) in the namespace of your type.
                using detail::sincos;
                return sincos(arg, result_sin, result_cos);
            }
        };

        //! The sqrt trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Sqrt
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sqrt(TArg) in the namespace of your type.
                using std::sqrt;
                return sqrt(arg);
            }
        };

        //! The tan trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Tan
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find tan(TArg) in the namespace of your type.
                using std::tan;
                return tan(arg);
            }
        };

        //! The trunc trait.
        template<typename T, typename TArg, typename TSfinae = void>
        struct Trunc
        {
            ALPAKA_FN_HOST_ACC auto operator()(T const& /* ctx */, TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find trunc(TArg) in the namespace of your type.
                using std::trunc;
                return trunc(arg);
            }
        };
    } // namespace traits

    //! Computes the absolute value.
    //!
    //! \tparam T The type of the object specializing Abs.
    //! \tparam TArg The arg type.
    //! \param abs_ctx The object specializing Abs.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto abs(T const& abs_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAbs, T>;
        return traits::Abs<ImplementationBase, TArg>{}(abs_ctx, arg);
    }

    //! Computes the principal value of the arc cosine.
    //!
    //! The valid real argument range is [-1.0, 1.0]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam TArg The arg type.
    //! \param acos_ctx The object specializing Acos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto acos(T const& acos_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAcos, T>;
        return traits::Acos<ImplementationBase, TArg>{}(acos_ctx, arg);
    }

    //! Computes the principal value of the arc sine.
    //!
    //! The valid real argument range is [-1.0, 1.0]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam TArg The arg type.
    //! \param asin_ctx The object specializing Asin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto asin(T const& asin_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAsin, T>;
        return traits::Asin<ImplementationBase, TArg>{}(asin_ctx, arg);
    }

    //! Computes the principal value of the arc tangent.
    //!
    //! \tparam TArg The arg type.
    //! \param atan_ctx The object specializing Atan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto atan(T const& atan_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAtan, T>;
        return traits::Atan<ImplementationBase, TArg>{}(atan_ctx, arg);
    }

    //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
    //!
    //! \tparam T The type of the object specializing Atan2.
    //! \tparam Ty The y arg type.
    //! \tparam Tx The x arg type.
    //! \param atan2_ctx The object specializing Atan2.
    //! \param y The y arg.
    //! \param x The x arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename Ty, typename Tx>
    ALPAKA_FN_HOST_ACC auto atan2(T const& atan2_ctx, Ty const& y, Tx const& x)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAtan2, T>;
        return traits::Atan2<ImplementationBase, Ty, Tx>{}(atan2_ctx, y, x);
    }

    //! Computes the cbrt.
    //!
    //! \tparam T The type of the object specializing Cbrt.
    //! \tparam TArg The arg type.
    //! \param cbrt_ctx The object specializing Cbrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto cbrt(T const& cbrt_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCbrt, T>;
        return traits::Cbrt<ImplementationBase, TArg>{}(cbrt_ctx, arg);
    }

    //! Computes the smallest integer value not less than arg.
    //!
    //! \tparam T The type of the object specializing Ceil.
    //! \tparam TArg The arg type.
    //! \param ceil_ctx The object specializing Ceil.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto ceil(T const& ceil_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCeil, T>;
        return traits::Ceil<ImplementationBase, TArg>{}(ceil_ctx, arg);
    }

    //! Computes the cosine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Cos.
    //! \tparam TArg The arg type.
    //! \param cos_ctx The object specializing Cos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto cos(T const& cos_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCos, T>;
        return traits::Cos<ImplementationBase, TArg>{}(cos_ctx, arg);
    }

    //! Computes the error function of arg.
    //!
    //! \tparam T The type of the object specializing Erf.
    //! \tparam TArg The arg type.
    //! \param erf_ctx The object specializing Erf.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto erf(T const& erf_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathErf, T>;
        return traits::Erf<ImplementationBase, TArg>{}(erf_ctx, arg);
    }

    //! Computes the e (Euler's number, 2.7182818) raised to the given power arg.
    //!
    //! \tparam T The type of the object specializing Exp.
    //! \tparam TArg The arg type.
    //! \param exp_ctx The object specializing Exp.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto exp(T const& exp_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathExp, T>;
        return traits::Exp<ImplementationBase, TArg>{}(exp_ctx, arg);
    }

    //! Computes the largest integer value not greater than arg.
    //!
    //! \tparam T The type of the object specializing Floor.
    //! \tparam TArg The arg type.
    //! \param floor_ctx The object specializing Floor.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto floor(T const& floor_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathFloor, T>;
        return traits::Floor<ImplementationBase, TArg>{}(floor_ctx, arg);
    }

    //! Computes the floating-point remainder of the division operation x/y.
    //!
    //! \tparam T The type of the object specializing Fmod.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param fmod_ctx The object specializing Fmod.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto fmod(T const& fmod_ctx, Tx const& x, Ty const& y)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathFmod, T>;
        return traits::Fmod<ImplementationBase, Tx, Ty>{}(fmod_ctx, x, y);
    }

    //! Checks if given value is finite.
    //!
    //! \tparam T The type of the object specializing Isfinite.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isfinite.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto isfinite(T const& ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsfinite, T>;
        return traits::Isfinite<ImplementationBase, TArg>{}(ctx, arg);
    }

    //! Checks if given value is inf.
    //!
    //! \tparam T The type of the object specializing Isinf.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isinf.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto isinf(T const& ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsinf, T>;
        return traits::Isinf<ImplementationBase, TArg>{}(ctx, arg);
    }

    //! Checks if given value is NaN.
    //!
    //! \tparam T The type of the object specializing Isnan.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isnan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto isnan(T const& ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsnan, T>;
        return traits::Isnan<ImplementationBase, TArg>{}(ctx, arg);
    }

    //! Computes the the natural (base e) logarithm of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Log.
    //! \tparam TArg The arg type.
    //! \param log_ctx The object specializing Log.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto log(T const& log_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathLog, T>;
        return traits::Log<ImplementationBase, TArg>{}(log_ctx, arg);
    }

    //! Returns the larger of two arguments.
    //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
    //!
    //! \tparam T The type of the object specializing Max.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param max_ctx The object specializing Max.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto max(T const& max_ctx, Tx const& x, Ty const& y)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathMax, T>;
        return traits::Max<ImplementationBase, Tx, Ty>{}(max_ctx, x, y);
    }

    //! Returns the smaller of two arguments.
    //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
    //!
    //! \tparam T The type of the object specializing Min.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param min_ctx The object specializing Min.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto min(T const& min_ctx, Tx const& x, Ty const& y)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathMin, T>;
        return traits::Min<ImplementationBase, Tx, Ty>{}(min_ctx, x, y);
    }

    //! Computes the value of base raised to the power exp.
    //!
    //! Valid real arguments for base are non-negative. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam T The type of the object specializing Pow.
    //! \tparam TBase The base type.
    //! \tparam TExp The exponent type.
    //! \param pow_ctx The object specializing Pow.
    //! \param base The base.
    //! \param exp The exponent.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TBase, typename TExp>
    ALPAKA_FN_HOST_ACC auto pow(T const& pow_ctx, TBase const& base, TExp const& exp)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathPow, T>;
        return traits::Pow<ImplementationBase, TBase, TExp>{}(pow_ctx, base, exp);
    }

    //! Computes the IEEE remainder of the floating point division operation x/y.
    //!
    //! \tparam T The type of the object specializing Remainder.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param remainder_ctx The object specializing Max.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto remainder(T const& remainder_ctx, Tx const& x, Ty const& y)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRemainder, T>;
        return traits::Remainder<ImplementationBase, Tx, Ty>{}(remainder_ctx, x, y);
    }

    //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from
    //! zero, regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param round_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto round(T const& round_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
        return traits::Round<ImplementationBase, TArg>{}(round_ctx, arg);
    }
    //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
    //! regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param lround_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto lround(T const& lround_ctx, TArg const& arg) -> long int
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
        return traits::Lround<ImplementationBase, TArg>{}(lround_ctx, arg);
    }
    //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
    //! regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param llround_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto llround(T const& llround_ctx, TArg const& arg) -> long long int
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
        return traits::Llround<ImplementationBase, TArg>{}(llround_ctx, arg);
    }

    //! Computes the rsqrt.
    //!
    //! Valid real arguments are positive. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Rsqrt.
    //! \tparam TArg The arg type.
    //! \param rsqrt_ctx The object specializing Rsqrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto rsqrt(T const& rsqrt_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRsqrt, T>;
        return traits::Rsqrt<ImplementationBase, TArg>{}(rsqrt_ctx, arg);
    }

    //! Computes the sine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Sin.
    //! \tparam TArg The arg type.
    //! \param sin_ctx The object specializing Sin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto sin(T const& sin_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSin, T>;
        return traits::Sin<ImplementationBase, TArg>{}(sin_ctx, arg);
    }

    //! Computes the sine and cosine (measured in radians).
    //!
    //! \tparam T The type of the object specializing SinCos.
    //! \tparam TArg The arg type.
    //! \param sincos_ctx The object specializing SinCos.
    //! \param arg The arg.
    //! \param result_sin result of sine
    //! \param result_cos result of cosine
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto sincos(T const& sincos_ctx, TArg const& arg, TArg& result_sin, TArg& result_cos) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSinCos, T>;
        traits::SinCos<ImplementationBase, TArg>{}(sincos_ctx, arg, result_sin, result_cos);
    }


    //! Computes the square root of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Sqrt.
    //! \tparam TArg The arg type.
    //! \param sqrt_ctx The object specializing Sqrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto sqrt(T const& sqrt_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSqrt, T>;
        return traits::Sqrt<ImplementationBase, TArg>{}(sqrt_ctx, arg);
    }

    //! Computes the tangent (measured in radians).
    //!
    //! \tparam T The type of the object specializing Tan.
    //! \tparam TArg The arg type.
    //! \param tan_ctx The object specializing Tan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto tan(T const& tan_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathTan, T>;
        return traits::Tan<ImplementationBase, TArg>{}(tan_ctx, arg);
    }

    //! Computes the nearest integer not greater in magnitude than arg.
    //!
    //! \tparam T The type of the object specializing Trunc.
    //! \tparam TArg The arg type.
    //! \param trunc_ctx The object specializing Trunc.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T, typename TArg>
    ALPAKA_FN_HOST_ACC auto trunc(T const& trunc_ctx, TArg const& arg)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMathTrunc, T>;
        return traits::Trunc<ImplementationBase, TArg>{}(trunc_ctx, arg);
    }
} // namespace alpaka::math
#include <iostream>
#include <cstdint>
#include <string>
#include <sstream>

#include <limits>
#include <cmath>

// GENERALIZATION
//
// E.g. fp16 is IEEEConverter<uint16_t, 5, 10, uint64_t>
template <typename T, int E, int M>
struct IEEEConverter {
    uint64_t  to_uint64(T v);
    T         from_u64(uint64_t v);
// extra methods
//  std::string source_to_binary(uint64_t)
};
//
// TODO: could generalize this to a base type as any type that supports >> &  and ==
//       could create two converters: one for either way
//       <typename S, unsigned S_E, unsigned S_M, typename T, unsigned T_E, unsigned T_M>
//
//
// Return type can be a status that tells us
//
// fp16:
// IEEEConverter<uint16_t, 5, 10, uint64_t> fp16;
//
// or use them as types
//
// template <typename T, int E, int M>
// struct IEEEFloat {
//    enum ConvertResult{
//      OK,         // norm -> norm
//                  // nan -> nan
//      OVERFLOW,   // * -> +-inf
//      UNDERFLOW,  // * -> +-0
//      ROUNDED,    // precision loss
//      NAN_LOSS,   // nan loses exact payload value (retains qnan,snan)
//      NAN_OK      // nan okay
//    };
//
//    T value; // must support >> << & | == !=
//
//    template <typename TT, int TE, int TM>
//    ConvertResult convertTo(IEEEFloat<TT,TE,TM> &target) const;
//
//    // e.g. typedef IEEEFloat<uint16_t,10,5> half_t;
//    //      auto x = half_t::createFrom<float>(f);
//    template<typename T>
//    static ConvertResult convertFrom(const T &v, IEEEFloat<T,E,M> &target);
//
//    // TODO:
//    // IEEEFloat<T,E,M> operator+(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator-(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator*(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator/(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator==(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator!=(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator>=(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator>(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator<(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator<=(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> operator-() const;
//    // IEEEFloat<T,E,M> fmod(const IEEEFloat<T,E,M> &rhs) const;
//    // IEEEFloat<T,E,M> sin(IEEEFloat<T,E,M> &val) const;
//    // IEEEFloat<T,E,M> cos(IEEEFloat<T,E,M> &val) const;
//    // ...
//
//    // static bool isnan(IEEEFloat<T,E,M> &val) { ... }
//    // static bool isqnan(IEEEFloat<T,E,M> &val) { ... }
//    // static bool issnan(IEEEFloat<T,E,M> &val) { ... }
//    // static bool isinf(IEEEFloat<T,E,M> &val) { ... }
//    // static bool iszero(IEEEFloat<T,E,M> &val) { ... } // e.g. ==0.0 (or -0.0)
//    // static bool isnegative(IEEEFloat<T,E,M> &val) { ... } // e.g <0.0 or ==-inf (not nan)
//    // ...
// };
//


//////////////////////////////////
//
// Implementation
//
static uint32_t f2u(float x) {
    union {
     float    f;
     uint32_t u;
    } bits;
    bits.f = x;
    return bits.u;
}

static float u2f(uint32_t x) {
    union {
     float    f;
     uint32_t u;
    } bits;
    bits.u = x;
    return bits.f;
}

#define FP16_MASK(X) ((1<<(X)) - 1)
// X bits starting at N
#define FP16_MASK_SHIFTED(X,N) (FP16_MASK(X) << N)
static const uint32_t FP16_EXP_MASK = FP16_MASK_SHIFTED(5,10); // bits[10..14] of fp16
static const uint16_t FP16_EXP_MAX = 0x1F;
static const uint32_t FP16_MAN_MASK = FP16_MASK(10); // bits[0..9] of fp16
static const uint32_t FP16_32_EXP_SHIFT = (22 - 10); // how much to shift to get at the exp bits
static const uint32_t FP16_BIAS = (1 << (5 - 1)) - 1; // 2^(e-1) - 1
static inline uint16_t FP16_EXP(uint16_t val) {
    if (val > FP16_EXP_MAX) {
        std::cerr << "FP16_EXP: out of bounds\n";
        abort();
    }
    return val << 10;
}

static const uint32_t FP32_EXP_MAX = 0xFF;
static const uint32_t FP32_EXP_MASK = FP16_MASK_SHIFTED(8,23);
static const uint32_t FP32_MAN_MAX = FP16_MASK(10);
static const uint32_t FP32_MAN_MASK = FP32_MAN_MAX;

static const uint32_t FP32_BIAS = 127;
static inline uint32_t FP32_EXP(uint32_t val) {
    if (val > FP32_EXP_MAX) {
        std::cerr << "FP32_EXP: out of bounds\n";
        abort();
    }
    return val << 23;
}

std::string format_fp16(uint16_t val);
std::string format_fp32(uint32_t val);

static uint32_t   fp16to32(uint16_t u16)
{
    std::cout << "fp16to32: " << format_fp16(u16) << "\n";
    uint32_t s32 = ((uint32_t)u16 & 0x8000) << (31 - 15); // bit [15 to 31]
    if ((u16 & FP16_EXP_MASK) == FP16_EXP_MASK) { // exp all 1's
        if ((u16 & FP16_MAN_MASK) != 0) { // NaN
            // strip the {q,s}high bit of the nan payload
            // e.g. S    11111              XYYYYYYYYY
            //                   /----------||||||||||
            // translates to    /            |||||||||
            //      S 11111111 X0000000000000YYYYYYYYY
            // this preserves the type of NaN (X) but retains the payload
            // in the low part of the word
            //
            const uint32_t QNAN_MASK = ((uint32_t)0x8000 >> (5+1));
            uint32_t m32_nan =
                (QNAN_MASK & u16) << (23 - 10) | ((~QNAN_MASK & ~(FP16_EXP_MASK|0x8000)) & u16);
            // we know at least one of these bits is non-zero
std::cout << "fp16to32: *nanh\n";
std::cout << "  M: " << std::hex << m32_nan << "\n";
std::cout << "    signaling: " << ((QNAN_MASK & u16) << (23 - 10)) <<  "\n";
std::cout << "  E: " << std::hex << FP32_EXP_MASK << "\n";
std::cout << "  S: " << std::hex << s32 << "\n";

            return s32 | FP32_EXP_MASK | m32_nan;
        } else {
std::cout << "fp16to32: +-inf\n";
            // +-Infinity: S 11111111 00000000000000000000000
            return s32 | FP32_EXP_MASK;
        }
    } else if ((u16 & FP16_EXP_MASK) == 0) {
        uint32_t u32_man = (uint32_t)(u16 & FP16_MAN_MASK);
        if (u32_man == 0) {
            // +-0
    std::cout << "fp16to32: +-0.0h\n";
            return s32;
        } else {
    std::cout << "fp16to32: den: m: " << std::hex << u32_man << "\n";
            // denorm fp16, we can convert this to a normalized fp32
            // by shifting the mantissa bits over one (we know we have
            // at least one more mantissa bit in fp32) and adding one
            // to the exponent
            // ==> (-1)^s * 2^-14 * 0.m...
            uint32_t m32_den = u32_man >> 1; // normalize

            return s32 | FP32_EXP(1) | m32_den;
        }
    } else {
        // norm
        // ==> (-1)^s * 2^-15 * 1.m...
        //
        // minimizes shifts: shift the biased bits up to fp32 location
        // and fix the bias -15 (for fp16, )
        //
        // fp32:                 s eeeee mmmmmmmmmm
        //         0000 -----------/////
        //         ||||/////
        // fp16: s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm
std::cout << std::hex << "fp16to32: norm\n";
        uint32_t e32_norm =
            ((uint32_t)u16 & FP16_EXP_MASK) +
            FP32_EXP(-FP16_BIAS + FP32_BIAS);
        // cannot overflow exponent bits

        uint32_t m32 = (uint32_t)(u16 & FP16_MAN_MASK);
        return s32 | e32_norm | m32;
    }
}


static uint16_t   fp32to16(uint32_t u32)
{
    uint16_t s16 = (uint16_t)((0x80000000 & u32) >> 16); // shift sign bit into it's final slot
    int32_t e32 = (int32_t)((0xFF & u32) >> 23) - 127; // unbias the exponent
    uint16_t m16 = (uint16_t)((FP32_MAN_MASK & u32) >> (22 - 10)); // top bits of the mantissa (bottom shifts off)
    std::cout << "fp32to16: " << format_fp32(u32) << "\n";

    if ((u32 & FP32_EXP_MASK) == FP32_EXP_MASK) {
        // all exp bits set
        if ((u32 & FP32_MAN_MASK) != 0) {
std::cout << "fp32to16: nan\n";
            // NaN try and preserve the NaN payload if possible
            // Note from IEE 754: any non-zero value in the mantissa means
            // NaN here.  A leading high bit means qnan, where a zero there
            // means a signalling NaN.
            //
            // We take the first 10 bits (this saves the qnan bit) truncating
            // the rest and then ensure the result is not zero (if the non-zero
            // part of m32 is in the truncated part we just set the low bit).
            uint16_t m16_nan =
                (uint16_t)((u32 & FP32_MAN_MASK) >> (22-10));
            if (m16 == 0) // ensure at least one bit is set in the mantissa so it doesn't become an infinity
                m16++;
            return (s16<<31) | FP16_EXP(FP16_EXP_MAX) | m16;
        } else {
std::cout << "fp32to16: inf\n";
            // +-Infinity
            return s16 | FP16_EXP(FP16_EXP_MAX);
        }
    } else if ((u32 & FP32_EXP_MASK) == 0) {
        // zero's in the exponent
        if ((FP32_MAN_MASK & u32) == 0) {
std::cout << "fp32to16: +/- 0.0h\n";
            // +0.0h/-0.0h
            return s16;
        } else {
            // denorms
            abort();
        }
    } else {
std::cout << "fp32to16: norm\n";
        // normal case
        // maybe round to denorm

        abort();
    }
}

uint16_t   fp32to16(float f32)
{
    return fp32to16(f2u(f32));
}

//////////////////////////////////
//
// Testing
//

// takes biased exponent values
static inline uint16_t fp16(uint16_t s, uint16_t e, uint16_t m)
{
    if (s > 1) {
        std::cerr << "fp16: sign bit invalid\n";
        abort();
    }
    if (e > 0x1F) {
        std::cerr << "fp16: expn bit invalid\n";
        abort();
    }
    if (m > 0x3FF) {
        std::cerr << "fp16: mant bit invalid\n";
        abort();
    }
    return (s << 15) | (e << 10) | m;
}
static inline uint32_t fp32(uint32_t s, uint32_t e, uint32_t m)
{
    if (s > 1) {
        std::cerr << "fp32: sign bit invalid\n";
        abort();
    }
    if (e > 0xFF) {
        std::cerr << "fp32: expn bit invalid\n";
        abort();
    }
    if (m > 0x7FFFFF) {
        std::cerr << "fp32: mant bit invalid\n";
        abort();
    }
    return (s << 31) | (e << 23) | m;
}


std::string format_fp16(uint16_t val)
{
    std::stringstream ss;
    auto fmtBit = [&](int i) {
        int bit = (val & (1 << i)) ? 1 : 0;
        ss << bit;
    };
    fmtBit(15);
    ss << ' ';
    for (int i = 14; i >= 10; i--)
        fmtBit(i);
    ss << ' ';
    for (int i = 9; i >= 0; i--)
        fmtBit(i);

    return ss.str();
}

std::string format_fp32(uint32_t val)
{
    std::stringstream ss;
    auto fmtBit = [&](int i) {
        int bit = (val & (1 << i)) ? 1 : 0;
        ss << bit;
    };
    fmtBit(31);
    ss << ' ';
    for (int i = 30; i >= 23; i--)
        fmtBit(i);
    ss << ' ';
    for (int i = 22; i >= 0; i--)
        fmtBit(i);

    return ss.str();
}

// static int fails;
static int totalFails;

// round trip test
//    fp32to16 . fp16to32 == id
//
static void test_rt_f16(const char *context, uint16_t val16, uint16_t expected_value)
{
    float mid = u2f(fp16to32(val16));
    std::cout << "====> MID =" << format_fp32(f2u(mid)) << "  " << mid << "\n";
    uint16_t rt_val16 = fp32to16(mid);
    std::cout << "<==== MID =" << format_fp16(rt_val16) << "\n";

    if (rt_val16 != expected_value) {
        if (context)
            std::cout << "  " << context << "\n";
        std::cout << "  error: " << format_fp16(val16) << " => " << format_fp32(f2u(mid)) << " (" << mid << ") => " << format_fp16(rt_val16) << "\n";
        std::cout << "    should be " << format_fp16(expected_value) << "\n";
        totalFails++;
    }
}
static void test_rt_f16(const char *context, uint16_t val16)
{
    test_rt_f16(context, val16, val16);
}



static void test16to32(const char *context, uint16_t in, uint32_t expected)
{
    auto res = fp16to32(in);
    if (res != expected) {
        std::cout << "TEST " << context << "\n";
        std::cout << std::hex << format_fp16(in) << " ==> " << format_fp32(res) << " (" << u2f(res) << ")\n";
        std::cout <<  "  expected             "        << format_fp32(expected) << " (" << u2f(expected) << ")\n";
        totalFails++;
    }
}
static void test32to16(const char *context, uint16_t in, float expected)
{
    test16to32(context, in, f2u(expected));
}
static void test32to16(const char *context, uint32_t in, uint16_t expected) { }
static void test32to16(const char *context, float in, uint16_t expected)
{
    test16to32(context, f2u(in), expected);
}

// TODO: fp8
// TODO: generalize to n bits

static void startGroup(const char *nm)
{
    std::cout << "-------------- GROUP " << nm << " ---------------\n";
}
static void endGroup()
{
   // if (fails > 0)
   //     std::cout << fails << " FAILED\n";
   // else
   //     std::cout << "PASSED\n";
    std::cout << "\n";
//    totalFails += fails;
//    fails = 0;
}

static void default_tests()
{
/*
    startGroup("ZEROS");
    test_rt_f16("0.0h",0x0000); //  0.0h
    test_rt_f16("-0.0h",0x8000); // -0.0h
    endGroup();

    startGroup("INF");
    test_rt_f16("inf",0x7c00); //  inf
    test_rt_f16("-inf",0xfc00); // -inf
    endGroup();
*/
    startGroup("NAN");
//    test16to32("snanh",  0x7C01 , 0x7F800001);
//    test16to32("-snanh", 0xFC01 , 0xFF800001);
    test16to32("snanh(0x7)",  0x7c07 , 0x7F800000);
//    test16to32("-snanh(0x7)",  0xFc07 , 0xFF800000);

//    test16to32("qnanh",  0x7E01 , 0x7FC00000);
//    test16to32("-qnanh", 0xFC01 , 0x8F800000);
//    test16to32("qnanh(0x77)", 0x7FC00077, 0x7c77);

//    test_rt_f16("snan",0x7c01); //   snan
//    test_rt_f16("-snan",0xfc01); //  -snan
    endGroup();

    // TODO: examples from https://en.wikipedia.org/wiki/Half-precision_floating-point_format

    // denorms
    // norms
    // nans

    std::cout << "********************************************\n";
    if (totalFails > 0)
        std::cout << totalFails << " FAILED\n";
    else
        std::cout << "PASSED\n";

}

int main(int argc, char ** argv)
{
    default_tests();
/*
    // 32->16
    // NAN's, INF, ->0x7f800000, ->0xff800000
    //
    // 32 goes to DENORMS come back
*/

    return totalFails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;

}
#include <iostream>
#include <chrono>
#include <stack>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <iomanip>
#include <limits>
#include <cassert>
#include <utility>
#include <functional>
#include <fstream>

#define COMPETITION

#ifndef COMPETITION

//#define COLOR_BOARD
#define USE_THREADS
#define INCLUDE_FILE_SYSTEM

#endif

#ifdef INCLUDE_FILE_SYSTEM
#include <filesystem>
#endif
#ifdef USE_THREADS
#include <thread>
#endif

typedef uint_fast8_t uint8;
typedef unsigned int uint;

template <typename T>
using vector_stack = std::stack<T, std::vector<T>>;

#define nd_c [[nodiscard]] constexpr

inline std::ostream &operator<<(std::ostream &out, unsigned char c) {
    return out << (int)c;
}

namespace Utils {

    template <typename T>
    constexpr T ceilDivide(T a, T b) {
        return 1 + ((a - 1) / b);
    }

    template <typename OUT=char, typename IN>
    constexpr static inline OUT* r_cast(IN &in) {
        return reinterpret_cast<OUT*>(&in);
    }

    template <typename OUT=char, typename IN>
    constexpr static inline const OUT* r_cast_const(const IN &in) {
        return reinterpret_cast<const OUT*>(&in);
    }

    template <typename... Ts>
    void swapEndian(Ts&... args) {
        ((args = (args>>24) |
                 ((args<<8) & 0x00FF0000) |
                 ((args>>8) & 0x0000FF00) |
                 (args<<24)), ...);
    }

    template <typename T>
    constexpr T* initializeArray(std::initializer_list<T> init) {
        auto r = new T[init.size()];
        std::copy(init.begin(), init.end(), r);
        return r;
    }

    template <typename T>
    constexpr void printHardCodeArray(const T* arr, uint len, std::ostream &out, const std::string &type="T") {
        out << "Utils::initializeArray<"<<type<<">({";
        for (uint i = 0; i < len; ++i) {
            out << (float)arr[i] << ',';
        }
        out << "})" << std::defaultfloat;
    }

    struct Random;
    static Random* RNG{};
    static std::mt19937* RNG2{};

    struct Random final {
        uint seed;

        explicit Random(
                const uint64_t &seed=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        ): seed(seed) {}

        [[nodiscard]]
        inline uint nextInt() {
            seed = (214013u * seed + 2531011u);
            return (seed >> 16u) & 0x7FFFu;
        }

        [[nodiscard]]
        inline uint nextInt(uint range) {
            return nextInt() % range;
        }

        [[nodiscard]]
        inline bool nextBoolean() {
            return nextInt() & 1u;
        }

        inline friend std::ostream &operator<<(std::ostream &out, const Random &rand) {
            out << rand.seed;
            return out;
        }

        static inline void init() {
            RNG = new Random();
            std::cerr << *RNG << '\n';
            auto seed2 = std::random_device()();
            std::cerr << seed2 << '\n';
            RNG2 = new std::mt19937(seed2);
        }
    };

    class BitSet64 {
    public:
        uint64_t word = 0;

        constexpr BitSet64() = default;
        constexpr BitSet64(uint64_t word): word(word) {}

        nd_c bool get(uint index) const {
            return (1ull << index) & word;
        }

        constexpr BitSet64& orSet(uint index) {
            word |= (1ull << index);
            return *this;
        }

        constexpr void unset(uint index) {
            word &= ~(1ull << index);
        }

        nd_c uint count() const {
            return __builtin_popcountll(word);
        }

        constexpr BitSet64 operator|(const BitSet64 &o) const {
            return word | o.word;
        }

        constexpr BitSet64 operator&(const BitSet64 &o) const {
            return word & o.word;
        }

        constexpr explicit operator bool() const {
            return word;
        }

        constexpr BitSet64 &operator|=(const BitSet64 &o) {
            word |= o.word;
            return *this;
        }

        constexpr BitSet64 &operator<<=(uint x) {
            word <<= x;
            return *this;
        }

        constexpr BitSet64 &operator>>=(uint x) {
            word >>= x;
            return *this;
        }

        nd_c BitSet64 operator^(const BitSet64 &o) const {
            return word ^ o.word;
        }

        template <typename T>
        nd_c T sub(uint pos, uint count) const {
            return (T)((word >> pos) & ((1ull << count) - 1ull));
        }

        struct iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type        = uint;

            uint64_t a = 0;
            constexpr void operator++() {
                a ^= -a & a;
            }
            constexpr uint operator*() const {
                return __builtin_ctzll(a);
            }

            constexpr bool operator!=(const iterator &o) const {
                return a != o.a;
            }
        };

        nd_c iterator begin() const {
            return {word};
        }

        nd_c iterator end() const {
            return {0};
        }
    };

    template <uint N>
    class TwoBitArray {
    public:
        BitSet64 _words[ceilDivide(N << 1, 64u)];
    public:
        constexpr TwoBitArray() = default;

        nd_c uint get(uint index) const {
            return _words[index / 32].template sub<uint>(index % 32 * 2, 2);
        }

        constexpr void orSet(uint index, uint v) {
            _words[index / 32].word |= ((uint64_t)v << (index % 32 * 2));
        }

        constexpr void unset(uint index) {
            _words[index / 32].word &= ~(3ull << (index % 32 * 2));
        }

        constexpr void set(uint index, uint v) {
            unset(index);
            if (v) orSet(index, v);
        }

    };
}

namespace Utils::Math {
    typedef double floating_type;
    typedef uint32_t size_type;

    struct Shape2D {
        size_type height=1, width=1;

        nd_c size_type area() const {
            return height * width;
        }

        friend std::ostream &operator<<(std::ostream &out, const Shape2D &shape) {
            return out << "{" + std::to_string(shape.height) + ", " + std::to_string(shape.width) + "}";
        }
    };

    struct Shape3D {
        size_type height=1, width=1, depth=1;

        constexpr size_type volume() const {
            return height * width * depth;
        }

        friend std::ostream &operator<<(std::ostream &out, const Shape3D &shape) {
            return out << "{" + std::to_string(shape.height) + ", " + std::to_string(shape.width) + ", " + std::to_string(shape.depth) + "}";
        }
    };

    namespace Matrix {

        template <
                typename MAT_P,
                typename T_P = typename MAT_P::value_type,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static MAT_P &dotReference(
                const MAT_A &A,
                const MAT_B &B,
                MAT_P &product
        ) {
            for (size_type j = 0; j < B.columns(); ++j) {
                for (size_type i = 0; i < A.rows(); ++i) {
                    product(i, j) = T_P();
                    for (size_type index = 0; index < A.columns(); ++index) {
                        product(i, j) += (T_P)(A(i, index) * B(index, j));
                    }
                }
            }
            return product;
        }

        template <
                typename MAT_P,
                typename Function1,
                typename Function2,
                typename T_P = typename MAT_P::value_type,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static void dotIteration(
                const MAT_A &A,
                const MAT_B &B,
                MAT_P* product,
                Function1 f1,
                Function2 f2
        ) {
            for (size_type j = 0; j < B.columns(); ++j) {
                for (size_type i = 0; i < A.rows(); ++i) {
                    T_P t = T_P();
                    for (size_type index = 0; index < A.columns(); ++index) {
                        f1(i, j, index);
                        t += static_cast<T_P>(A(i, index) * B(index, j));
                    }
                    f2(t, i, j);
                    if (product) (*product)(i, j) = t;
                }
            }
        }

        template <
                typename MAT_P,
                typename T_P = typename MAT_P::value_type,
                typename... Args,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static MAT_P dot(
                const MAT_A &A,
                const MAT_B &B,
                Args&&... args
        ) {
            MAT_P product(std::forward<Args>(args)...);
            dotReference<MAT_P, T_P>(A, B, product);
            return product;
        }

        template <
                typename MAT_S,
                typename T_P = typename MAT_S::value_type,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static MAT_S &addReference(
                const MAT_A &A,
                const MAT_B &B,
                MAT_S &sum
        ) {
            for (size_type i = 0; i < A.rows(); ++i) for (size_type j = 0; j < A.columns(); ++j)
                    sum(i, j) = static_cast<T_P>(A(i, j) + B(i, j));
            return sum;
        }

        template <
                typename MAT_S,
                typename T_P = typename MAT_S::value_type,
                typename... Args,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static MAT_S add(
                const MAT_A &A,
                const MAT_B &B,
                Args&&... args
        ) {
            MAT_S sum(std::forward<Args>(args)...);
            addReference<MAT_S, T_P>(A, B, sum);
            return sum;
        }

        template <
                typename MAT_R,
                typename T_P = typename MAT_R::value_type,
                typename MAT_A,
                typename MAT_B
        >
        constexpr static void convolve(
                const MAT_A &A,
                const MAT_B &kernel,
                MAT_R &result
        ) {
            for (size_type ki = 0; ki < kernel.rows(); ++ki) {
                for (size_type kj = 0; kj < kernel.columns(); ++kj) {
                    auto kij = kernel(ki, kj);
                    if (!kij) continue;
                    for (size_type i = 0; i < result.rows(); ++i) { // stride with i += stride
                        for (size_type j = 0; j < result.columns(); ++j) {
                            result(i, j) += static_cast<T_P>(A(i+ki, j+kj) * kij);
                        }
                    }
                }
            }
        }

        template <typename MAT, typename T_M = typename MAT::value_type>
        constexpr static void fill(MAT &matrix, const T_M &a) {
            for (size_type i = 0; i < matrix.rows(); ++i) {
                for (size_type j = 0; j < matrix.columns(); ++j) {
                    matrix(i, j) = a;
                }
            }
        }

        template <typename R, typename MAT, typename... Args>
        constexpr static void copy(const MAT &m, Args&&... args) {
            R result(std::forward<Args>(args)...);
            for (size_type i = 0; i < m.rows(); ++i) {
                for (size_type j = 0; j < m.columns(); ++j) {
                    result(i, j) = m(i, j);
                }
            }
            return result;
        }

        template <typename MAT>
        std::ostream &print(const MAT &matrix, std::ostream &out=std::cout) {
            out << '{';
            for (size_type i = 0; i < matrix.rows(); ++i) {
                if (i) out << ' ';
                out << '{';
                for (size_type j = 0; j < matrix.columns(); ++j) {
                    out << (j? ',': ' ') << "  " << matrix(i, j);
                }
                out << '}' << (i+1 == matrix.rows()? '}': '\n');
            }
            return out << '\n';
        }

        template <typename MAT> class TransposedMatrixReference;

        template <typename MAT>
        class TransposedMatrixReference {
        public:
            typedef MAT Matrix;
            typedef typename Matrix::value_type value_type;

        protected:
            Matrix &ref;
        public:

            constexpr explicit TransposedMatrixReference(Matrix &matrix): ref(matrix) {
            }

            // Start Matrix Functions
            [[nodiscard]]
            constexpr size_type rows() const noexcept {
                return ref.columns();
            }

            [[nodiscard]]
            constexpr size_type columns() const noexcept {
                return ref.rows();
            }

            // Start Operators
            [[nodiscard]]
            constexpr value_type &operator()(size_type i, size_type j) {
                return ref(j, i);
            }

            [[nodiscard]]
            constexpr const value_type &operator()(size_type i, size_type j) const {
                return ref(j, i);
            }
            // End Operators
            // End Matrix Functions
        };

        template <typename MAT>
        class PaddedMatrixReference {
        public:
            typedef MAT Matrix;
            typedef typename Matrix::value_type value_type;

        protected:
            const Matrix &ref;
            const size_type padding;
            const value_type padding_fill;
        public:

            constexpr PaddedMatrixReference(const Matrix &matrix, size_type padding, value_type paddingFill=0):
                    ref(matrix), padding(padding), padding_fill(paddingFill)
            {}

            // Start Matrix Functions
            [[nodiscard]]
            constexpr size_type rows() const noexcept {
                return ref.rows() + (padding << 1);
            }

            [[nodiscard]]
            constexpr size_type columns() const noexcept {
                return ref.columns() + (padding << 1);
            }

            // Start Operators
            [[nodiscard]]
            constexpr const value_type &operator()(size_type i, size_type j) const {
                if (i-padding < ref.rows() && j-padding < ref.columns()) return ref(i-padding, j-padding);
                return padding_fill;
            }
            // End Operators
            // End Matrix Functions

        };

        template <typename T = floating_type>
        class DynamicMatrix {
        public:
            typedef T value_type;

            size_type m = 0, n = 0;
            value_type* array = nullptr;

            // Start Constructors
            constexpr DynamicMatrix() = default;

            template <typename InputIt>
            constexpr DynamicMatrix(InputIt first, InputIt last, size_type rows):
                    m(rows),
                    n(std::distance(first, last) / rows),
                    array(new value_type[std::distance(first, last)]) {
                std::copy(first, last, array);
            }
            constexpr explicit DynamicMatrix(size_type rows, size_type columns):
                    m(rows),
                    n(columns),
                    array(new value_type[rows * columns]) {
            }

            constexpr DynamicMatrix(const DynamicMatrix &o): DynamicMatrix(
                    o.array,
                    o.array + o.m * o.n,
                    o.m) {
                //std::cerr << "copied matrix\n";
            }

            constexpr DynamicMatrix(DynamicMatrix &&o) noexcept: m(std::move(o.m)), n(std::move(o.n)), array(o.array) {
                o.array = nullptr;
            }

            constexpr DynamicMatrix(std::initializer_list<std::initializer_list<value_type>> l):
                    m(l.size()), n(l.begin()->size())
            {
                array = new value_type[m * n];
                for (size_type i = 0; i < l.size(); ++i) {
                    const auto &itl = l.begin() + i;
                    assert(("All rows must be equal sized.", itl->size() == n));
                    std::copy<typename std::initializer_list<value_type>::const_iterator, value_type*>(
                            itl->begin(), itl->end(), array + i * n
                    );
                }
                std::cout << "created matrix " << this << '\n';
            }

            constexpr DynamicMatrix(value_type* arr, size_type m, size_type n): m(m), n(n), array(arr) {
            }
            // End Constructors

            constexpr DynamicMatrix &operator=(const DynamicMatrix &o) noexcept {
                std::cout << "copied matrix\n";
                delete[] array;
                m = o.m;
                n = o.n;
                array = new value_type[m * n];
                std::copy(o.array, o.array + m * n, array);
                //std::cout << "copied matrix " << &o << " to " << this << "\n";
                return *this;
            }

            constexpr DynamicMatrix &operator=(DynamicMatrix &&o) noexcept {
                // std::cout << "moved matrix\n";
                delete[] array;
                m = o.m;
                n = o.n;
                array = o.array;
                o.array = nullptr;
                return *this;
            }

        public:
            ~DynamicMatrix() {
                delete[] array;
            }

            // Start Matrix Functions

            [[nodiscard]]
            constexpr size_type rows() const noexcept {
                return m;
            }

            [[nodiscard]]
            constexpr size_type columns() const noexcept {
                return n;
            }

            // Start Operators
            [[nodiscard]]
            constexpr value_type &operator()(size_type i, size_type j) {
                return array[i * n + j];
            }

            [[nodiscard]]
            constexpr const value_type &operator()(size_type i, size_type j) const {
                return array[i * n + j];
            }
            // End Operators
            // End Matrix Functions

            template <
                    typename T_O = value_type,
                    typename T_R=T_O
            >
            constexpr DynamicMatrix<T_R> operator*(const DynamicMatrix<T_O> &other) const {
                return dot<DynamicMatrix<T_R>, T_R>(*this, other, m, other.n);
            }

            inline friend std::ostream &operator<<(std::ostream &out, const DynamicMatrix &matrix) {
                return print(matrix);
            }
        };

        template <typename T = floating_type, bool IS_COLUMN_VECTOR=true>
        class DynamicVector {
        public:
            typedef T value_type;

            static constexpr const bool vector_type = IS_COLUMN_VECTOR;

            size_type n = 0;
            value_type* array = nullptr;

            // Start Constructors
            constexpr DynamicVector() = default;

            template <typename InputIt>
            constexpr DynamicVector(InputIt first, InputIt last):
                    n(std::distance(first, last)),
                    array(new value_type[std::distance(first, last)]) {
                std::copy(first, last, array);
            }
            constexpr explicit DynamicVector(size_type _n):
                    n(_n),
                    array(new value_type[_n]) {
            }

            constexpr DynamicVector(const DynamicVector &o): DynamicVector(o.array, o.array + o.n) {
                std::cerr << "copied vector\n";
            }

            constexpr DynamicVector(DynamicVector &&o) noexcept: n(std::move(o.n)), array(o.array) {
                o.array = nullptr;
            }

            constexpr DynamicVector(std::initializer_list<value_type> l):
                    DynamicVector(l.begin(), l.end()) {
            }

            constexpr DynamicVector(value_type* arr, size_type n): n(n), array(arr) {
            }
            // End Constructors

            constexpr DynamicVector &operator=(const DynamicVector &o) noexcept {
                std::cout << "copied vector\n";
                delete[] array;
                n = o.n;
                array = new value_type[n];
                std::copy(o.array, o.array + n, array);
                return *this;
            }

            constexpr DynamicVector &operator=(DynamicVector &&o) noexcept {
                //std::cout << "moved vector\n";
                delete[] array;
                n = o.n;
                array = o.array;
                o.array = nullptr;
                return *this;
            }

        public:
            ~DynamicVector() {
                delete[] array;
            }

            // Start Matrix Functions

            [[nodiscard]]
            constexpr size_type rows() const noexcept {
                return vector_type? n: 1;
            }

            [[nodiscard]]
            constexpr size_type columns() const noexcept {
                return vector_type? 1: n;
            }

            // Start Operators
            [[nodiscard]]
            constexpr value_type &operator()(size_type i, size_type j) {
                return array[i + j];
            }

            [[nodiscard]]
            constexpr const value_type &operator()(size_type i, size_type j) const {
                return array[i + j];
            }
            // End Operators
            // End Matrix Functions

            [[nodiscard]]
            constexpr value_type &operator[](size_type i) {
                return array[i];
            }

            [[nodiscard]]
            constexpr const value_type &operator[](size_type i) const {
                return array[i];
            }

            inline friend std::ostream &operator<<(std::ostream &out, const DynamicVector &matrix) {
                return print(matrix, out);
            }
        };
    }

    template <typename T>
    Matrix::DynamicVector<T> dirichlet(T alpha, uint size, std::mt19937 &gen) {
        Matrix::DynamicVector<T> result(size);
        std::gamma_distribution<> gamma(alpha, 1);
        T sum = 0;
        for (uint i = 0; i < size; ++i) sum += (result[i] = gamma(gen));
        for (uint i = 0; i < size; ++i) result[i] /= sum;
        return result;
    }
}

namespace DeepLearning {
    using namespace Utils::Math;
    using namespace Utils::Math::Matrix;

    template <typename T = floating_type>
    class Network;

    template <typename T = floating_type>
    class LayerBlock;

    namespace Optimizers {
        template <typename T>
        struct Optimizer {
            typedef T value_type;

            virtual ~Optimizer() = default;

            virtual void updateSingleWeight(value_type &w, value_type grad) = 0;
            virtual void preUpdate() = 0;
            virtual void postEpoch() = 0;
            virtual void printStats(std::ostream &out) const = 0;
        };

        template <typename T>
        struct SGD : public Optimizer<T> {
            typedef Optimizer<T> Base;
            typedef typename Base::value_type value_type;

            double learning_rate, decay;

            explicit SGD(double learning_rate, double decay_factor=1.0): learning_rate(learning_rate), decay(decay_factor) {}

            SGD(const SGD &o) = default;

            ~SGD() override {
                this->printStats(std::cerr);
            }

            void updateSingleWeight(value_type &w, value_type grad) override {
                w -= grad*learning_rate;
            }

            void postEpoch() override {
                learning_rate *= decay;
            }

            void preUpdate() override {
            }

            void printStats(std::ostream &out) const override {
                out << "SGD Destructed:\n learning rate: " << learning_rate << ", decay: " << decay << std::endl << std::defaultfloat;
            }
        };

        template <typename T>
        struct Adam : public Optimizer<T> {
            typedef Optimizer<T> Base;
            typedef typename Base::value_type value_type;

            double learning_rate, beta1, beta2, eps;
            const double init_lr;

            value_type* m,* v;

            uint updates = 0, currentParam = 0;

            explicit Adam(size_type totalParams, double learning_rate=.001, double beta1=.9, double beta2=.999, double eps=1e-7):
                    learning_rate(learning_rate), beta1(beta1), beta2(beta2), eps(eps), init_lr(learning_rate) {
                m = new value_type[totalParams*2];
                v = m + totalParams;
                std::fill(m, m+totalParams*2, 0.0);
            }

            ~Adam() override {
                delete[] m;
                this->printStats(std::cerr);
            }

            void updateSingleWeight(value_type &w, value_type grad) override {
                m[currentParam] = beta1 * m[currentParam] + (1.0 - beta1) * grad;
                v[currentParam] = beta2 * v[currentParam] + (1.0 - beta2) * grad * grad;

                w -= learning_rate * m[currentParam] / (std::sqrt(v[currentParam]) + eps);

                ++currentParam;
            }

            void postEpoch() override {
            }

            void preUpdate() override {
                currentParam = 0;
                if (updates)
                    learning_rate = init_lr * sqrt(1.0 - std::pow(beta2, updates)) / (1.0 - std::pow(beta1, updates));
                ++updates;
            }

            void printStats(std::ostream &out) const override {
                out << "Adam Destructed:\n updates: " << updates << ", initial learning rate: ";
                out << init_lr << ", current learning rate: ";
                out << learning_rate << ", beta1: ";
                out << beta1 << ", beta2: ";
                out << beta2 << ", eps: ";
                out << eps << std::endl;
                out << std::defaultfloat;
            }
        };
    }

    namespace Activation {
        enum ActivationType: uint {
            SIGMOID = 1,
            RELU = 2,
            SOFTMAX = 3,
            TANH = 4
        };

        template<
                typename T
        >
        struct Sigmoid {
            typedef T value_type;
            typedef value_type* Data;

            constexpr static value_type sigmoid(value_type x) {
                return 1.0 / (1.0 + std::exp(-x));
            }

            constexpr static void f(Data in, Data out, size_type n) {
                std::transform(in, in + n, out, sigmoid);
            }

            constexpr static void fd(Data out, Data deltaIn, Data deltaOut, size_type n) {
                for (size_type i = 0; i < n; ++i) deltaOut[i] = out[i] * (1.0 - out[i]) * deltaIn[i];
            }
        };

        template<
                typename T
        >
        struct Tanh {
            typedef T value_type;
            typedef value_type* Data;

            constexpr static void f(Data in, Data out, size_type n) {
                for (size_type i = 0; i < n; ++i) out[i] = std::tanh(in[i]);
            }

            constexpr static void fd(Data out, Data deltaIn, Data deltaOut, size_type n) {
                for (size_type i = 0; i < n; ++i) deltaOut[i] = (1.0 - out[i]*out[i]) * deltaIn[i];
            }
        };

        template<
                typename T
        >
        struct ReLU {
            typedef T value_type;
            typedef value_type* Data;

            constexpr static void f(Data in, Data out, size_type n) {
                for (size_type i = 0; i < n; ++i) out[i] = in[i] > 0.0? in[i]: 0.0;
            }

            constexpr static void fd(Data out, Data deltaIn, Data deltaOut, size_type n) {
                for (size_type i = 0; i < n; ++i) deltaOut[i] = out[i]? deltaIn[i]: 0.0;
            }
        };

        template<
                typename T
        >
        struct SoftMax {
            typedef T value_type;
            typedef value_type* Data;

            constexpr static void f(Data in, Data out, size_type n) {
                value_type sum = 0.0, max = *std::max_element(in, in+n);
                for (size_type i = 0; i < n; ++i) {
                    sum += (out[i] = std::exp(in[i] - max));
                }
                for (size_type i = 0; i < n; ++i) {
                    //std::cout << "out: " << out[i] << " = exp(" << in[i] << " - max)\n";
                    out[i] /= sum;
                }
            }

            constexpr static void fd(Data out, Data deltaIn, Data deltaOut, size_type n) {
                value_type jac;
                for (size_type i = 0; i < n; ++i) {
                    deltaOut[i] = 0.0;
                    if (!deltaIn[i]) continue;
                    for (size_type j = 0; j < n; ++j) {
                        jac = i == j? out[i] * (1.0 - out[i]): -out[i] * out[j];
                        deltaOut[i] += jac * out[j];
                    }
                    deltaOut[i] *= deltaIn[i];
                }
            }
        };

        template <typename T>
        static constexpr
        std::pair<
                std::function<void(T*, T*, size_type)>,
                std::function<void(T*, T*, T*, size_type)>
        > getActivation(ActivationType t) {
            switch (t) {
                case SIGMOID:
                    return {Sigmoid<T>::f, Sigmoid<T>::fd};
                case RELU:
                    return {ReLU<T>::f, ReLU<T>::fd};
                case SOFTMAX:
                    return {SoftMax<T>::f, SoftMax<T>::fd};
                case TANH:
                    return {Tanh<T>::f, Tanh<T>::fd};
                default:
                    return {nullptr, nullptr};
            }
        }

        static constexpr const char* getActivationName(ActivationType t) {
            switch (t) {
                case SIGMOID:
                    return "Sigmoid";
                case RELU:
                    return "ReLU";
                case SOFTMAX:
                    return "SoftMax";
                case TANH:
                    return "Tanh";
                default:
                    return "";
            }
        }
    }

    namespace Initializers {
        template <typename MAT, typename T_M = typename MAT::value_type>
        constexpr static void xavier(MAT &matrix, size_type fanIn, size_type fanOut, std::mt19937 &gen) {
            const T_M xv = 2.449489742783178 / std::sqrt(static_cast<T_M>(fanIn + fanOut));
            std::uniform_real_distribution<T_M> dis(-xv, xv);
            for (size_type i = 0; i < matrix.rows(); ++i) {
                for (size_type j = 0; j < matrix.columns(); ++j) {
                    matrix(i, j) = dis(gen);
                }
            }
        }
    }

    template <
            typename T = floating_type
    >
    class Layer {
    public:
        typedef T value_type;
        typedef value_type* Data;
        typedef DynamicMatrix<value_type> Matrix;
        typedef DynamicVector<value_type> Vector;

    protected:
        constexpr explicit Layer(const Shape3D &outputSize, size_type params, const char* name): output_size(outputSize), params(params), name(name) {}

        virtual void forward(Data, Data) const = 0;
        virtual void backward(Data, Data, Data, Data) const = 0;
        virtual void updateParameters(Optimizers::Optimizer<value_type> &optimizer) = 0;
        virtual void printHardCodeInitializer(std::ostream &) const = 0;
    public:
        const Shape3D output_size;
        const size_type params;
        const std::string name;

        virtual ~Layer() = default;

        virtual std::ofstream &save(std::ofstream&) const = 0;

        friend class Network<value_type>;
        friend class LayerBlock<value_type>;
    };

    enum LayerType : char {
        DENSE = 0,
        CONV2D = 1,
        ACTIVATION = 2,
        MAXPOOL2D = 3
    };

    template <typename T>
    class DenseLayer : public Layer<T> {
    public:
        typedef T value_type;
        typedef Layer<value_type> Base;

        typedef typename Base::Data Data;
        typedef typename Base::Matrix Matrix;
        typedef typename Base::Vector Vector;
    protected:
        Matrix weight{};
        Vector bias{};

    public:
        mutable Matrix _delta_weight;
        mutable Vector _delta_bias;

        constexpr DenseLayer(size_type inputSize, size_type outputSize, Data weights):
                Base({outputSize}, outputSize*inputSize+outputSize, "Dense"),
                _delta_weight(outputSize, inputSize),
                _delta_bias(outputSize)
        {
            weight = Matrix(weights, outputSize, inputSize);
            bias = Vector(weights + inputSize * outputSize, outputSize);
            fill(_delta_weight, 0);
            fill(_delta_bias, 0);

            /*std::cout << "loading dense\n";
            print(weight);
            print(bias);
            std::cout << inputSize << '\n';
            std::cout << outputSize << '\n';
            std::cout << Base::params << '\n';*/
        }

        constexpr DenseLayer(
                size_type inputSize,
                size_type outputSize,
                std::mt19937 &gen,
                const std::function<void(Matrix&, size_type, size_type, std::mt19937&)> &initializer=Initializers::xavier<Matrix>
        ):
                Base({outputSize}, outputSize*inputSize+outputSize, "Dense"),
                _delta_weight(outputSize, inputSize),
                _delta_bias(outputSize)
        {
            auto weights = new value_type[inputSize * outputSize + outputSize];
            weight = Matrix(weights, outputSize, inputSize);
            bias = Vector(weights + inputSize * outputSize, outputSize);

            initializer(weight, inputSize, outputSize, gen);
            fill(bias, 0);

            fill(_delta_weight, 0);
            fill(_delta_bias, 0);

            /*std::cout << "making dense\n";
            print(weight);
            print(bias);
            std::cout << inputSize << '\n';
            std::cout << outputSize << '\n';
            std::cout << Base::params << '\n';*/
        }

        ~DenseLayer() override {
            bias.array = nullptr;
        }

        void forward(Data _in, Data _out) const override {
            Vector in(_in, weight.columns());
            Vector out(_out, bias.rows());
            dotIteration(weight, in, &out,
                         [](size_type, size_type, size_type) noexcept {},
                         [this](value_type &p, size_type i, size_type j) noexcept {
                             p += this->bias[i];
                         }
            );
            in.array = out.array = nullptr;
        }

        void backward(Data _in, Data, Data _deltaIn, Data _deltaOut) const override {
            Vector in(_in, weight.columns()), deltaIn(_deltaIn, bias.rows());
            Vector deltaOut(_deltaOut, weight.columns());
            dotIteration(TransposedMatrixReference(weight), deltaIn, &deltaOut,
                         [this, &in, &deltaIn](size_type i, size_type j, size_type index) noexcept {
                             _delta_weight(index, i) += deltaIn[index] * in[i];
                             if (!i) _delta_bias[index] += deltaIn[index];
                         },
                         [](value_type, size_type, size_type) noexcept {
                         }
            );
            in.array = deltaIn.array = deltaOut.array = nullptr;
        }

        void updateParameters(Optimizers::Optimizer<value_type> &optimizer) override {
            for (size_type i = 0; i < weight.rows(); ++i) {
                optimizer.updateSingleWeight(bias[i], _delta_bias[i]);
                _delta_bias[i] = 0;
                for (size_type j = 0; j < weight.columns(); ++j) {
                    optimizer.updateSingleWeight(weight(i, j), _delta_weight(i, j));
                    _delta_weight(i, j) = 0;
                }
            }
        }

        std::ofstream &save(std::ofstream& out) const override {
            using Utils::r_cast_const;

            out.put(DENSE);

            auto size_in = weight.columns();
            auto size_out = bias.rows();
            out.write(r_cast_const(size_in), sizeof(size_type));
            out.write(r_cast_const(size_out), sizeof(size_type));

            out.write(r_cast_const(*weight.array), Base::params*sizeof(value_type));
            out.flush();
            return out;
        }

        constexpr static DenseLayer* load(std::ifstream& in) {
            using Utils::r_cast;

            size_type size_in=0, size_out=0;

            in.read(r_cast(size_in), sizeof(size_type));
            in.read(r_cast(size_out), sizeof(size_type));

            auto params = size_in * size_out + size_out;
            auto weights = new value_type[params];
            in.read(r_cast(*weights), params*sizeof(value_type));

            return new DenseLayer(size_in, size_out, weights);
        }

        void printHardCodeInitializer(std::ostream &out) const override {
            out << "new DenseLayer<T>("<<weight.columns()<<", "<<Base::output_size.volume()<<", ";
            Utils::printHardCodeArray(weight.array, Base::params, out);
            out << ')';
        }
    };

    template <typename T>
    class Conv2D : public Layer<T> {
    public:
        typedef T value_type;
        typedef Layer<value_type> Base;

        typedef typename Base::Data Data;
        typedef typename Base::Matrix Matrix;
        typedef typename Base::Vector Vector;
    protected:
        const Shape3D input_size;
        const Shape2D kernel_size;
        const size_type padding = 0;

        DynamicMatrix<Matrix> kernels;
        Vector bias{};
    public:
        mutable DynamicMatrix<Matrix> _delta_kernels;
        mutable Vector _delta_bias{};

        constexpr Conv2D(
                const Shape3D &input_size,
                const Shape2D &kernel_size,
                size_type planes_out,
                size_type padding,
                Data weights
        ): // <----- :)
                Base(
                        {
                                input_size.height - kernel_size.height + 1 + padding * 2,
                                input_size.width - kernel_size.width + 1 + padding * 2,
                                planes_out
                        }, planes_out*input_size.depth*kernel_size.area()+planes_out, "Conv2D"),
                input_size(input_size),
                kernel_size(kernel_size),
                padding(padding),
                kernels(planes_out, input_size.depth),
                bias(planes_out),
                _delta_kernels(planes_out, input_size.depth)
        {
            auto params = input_size.depth * planes_out * kernel_size.area() + planes_out;
            auto grad = new value_type[params];

            std::fill(grad, grad + params, 0);

            bias = Vector(weights + params - planes_out, planes_out);
            _delta_bias = Vector(grad + params - planes_out, planes_out);

            //std::cout << "loading conv\n";
            size_type ki = 0;
            for (size_type i = 0; i < planes_out; ++i) {
                for (size_type j = 0; j < input_size.depth; ++j) {
                    kernels(i, j) = Matrix(weights + ki * kernel_size.area(), kernel_size.height, kernel_size.width);
                    //print(kernels(i, j));
                    _delta_kernels(i, j) = Matrix(grad + ki * kernel_size.area(), kernel_size.height, kernel_size.width);
                    ++ki;
                }
            }
        }

        constexpr Conv2D(
                const Shape3D &input_size,
                const Shape2D &kernel_size,
                size_type planes_out,
                std::mt19937 &gen,
                size_type padding=0,
                const std::function<void(Matrix&, size_type, size_type, std::mt19937&)> &initializer=Initializers::xavier<Matrix>
        ):
                Base(
                        {
                                input_size.height - kernel_size.height + 1 + padding * 2,
                                input_size.width - kernel_size.width + 1 + padding * 2,
                                planes_out
                        }, planes_out*input_size.depth*kernel_size.area()+planes_out, "Conv2D"),
                input_size(input_size),
                kernel_size(kernel_size),
                padding(padding),
                kernels(planes_out, input_size.depth),
                _delta_kernels(planes_out, input_size.depth)
        {
            auto params = Base::params;
            auto weights = new value_type[params];
            auto grad = new value_type[params];

            std::fill(grad, grad + params, 0);

            bias = Vector(weights + params - planes_out, planes_out);
            _delta_bias = Vector(grad + params - planes_out, planes_out);

            size_type
                    fanIn = input_size.depth * kernel_size.width * kernel_size.height,
                    fanOut = input_size.depth * kernel_size.width * kernel_size.height;
            //std::cout << "making conv\n";
            size_type ki = 0;
            for (size_type i = 0; i < planes_out; ++i) {
                bias[i] = 0;
                for (size_type j = 0; j < input_size.depth; ++j) {
                    kernels(i, j) = Matrix(weights + ki * kernel_size.area(), kernel_size.height, kernel_size.width);
                    initializer(kernels(i, j), fanIn, fanOut, gen);
                    //print(kernels(i, j));

                    _delta_kernels(i, j) = Matrix(grad + ki * kernel_size.area(), kernel_size.height, kernel_size.width);

                    ++ki;
                }
            }
        }

        ~Conv2D() override {
            delete[] kernels(0, 0).array;
            delete[] _delta_kernels(0, 0).array;
            for (size_type i = 0; i < Base::output_size.depth; ++i) {
                for (size_type j = 0; j < input_size.depth; ++j) {
                    kernels(i, j).array = nullptr;
                    _delta_kernels(i, j).array = nullptr;
                }
            }
            bias.array = nullptr;
            _delta_bias.array = nullptr;
        }

        void forward(Data _in, Data _out) const override {
            size_type volume = Base::output_size.height * Base::output_size.width;

            Matrix out(_out, Base::output_size.height, Base::output_size.width);
            Matrix in(_in, input_size.height, input_size.width);
            for (size_type output_plane = 0; output_plane < kernels.rows(); ++output_plane) {
                out.array = _out + volume * output_plane;
                std::fill(out.array, out.array + volume, bias[output_plane]);
                for (size_type input_plane = 0; input_plane < kernels.columns(); ++input_plane) {
                    const Matrix &kernel = kernels(output_plane, input_plane);
                    in.array = _in + input_size.width * input_size.height * input_plane;
                    if (padding) convolve(PaddedMatrixReference(in, padding), kernel, out);
                    else convolve(in, kernel, out);
                }
            }
            out.array = nullptr;
            in.array = nullptr;
        }

        void backward(Data _in, Data, Data _deltaIn, Data _deltaOut) const override {
            size_type
                    output_width  = Base::output_size.width,
                    output_height = Base::output_size.height;

            std::fill(_deltaOut, _deltaOut + input_size.volume(), 0);
            for (size_type output_plane = 0; output_plane < kernels.rows(); ++output_plane) {
                Matrix deltaIn(_deltaIn + output_width * output_height * output_plane, output_height, output_width);
                for (size_type i = 0; i < output_height; ++i) {
                    for (size_type j = 0; j < output_width; ++j) {
                        value_type deltaInIJ = deltaIn(i, j);
                        if (!deltaInIJ) continue;
                        _delta_bias[output_plane] += deltaInIJ;
                        for (size_type input_plane = 0; input_plane < kernels.columns(); ++input_plane) {
                            Matrix in(
                                    _in + input_size.width * input_size.height * input_plane,
                                    input_size.height, input_size.width);
                            Matrix deltaOut(
                                    _deltaOut + input_size.width * input_size.height * input_plane,
                                    input_size.height, input_size.width);
                            for (size_type ki = 0; ki < kernel_size.height; ++ki) {
                                for (size_type kj = 0; kj < kernel_size.width; ++kj) {
                                    size_type y = i + ki - padding, x = j + kj - padding;
                                    if (y >= input_size.height || x >= input_size.width) continue;
                                    _delta_kernels(output_plane, input_plane)(ki, kj) += deltaInIJ * in(y, x);
                                    deltaOut(y, x) += deltaInIJ * kernels(output_plane, input_plane)(ki, kj);
                                }
                            }
                            deltaOut.array = in.array = nullptr;
                        }
                    }
                }
                deltaIn.array = nullptr;
            }
        }

        void updateParameters(Optimizers::Optimizer<value_type> &optimizer) override {
            for (size_type i = 0; i < kernels.rows(); ++i) {
                optimizer.updateSingleWeight(bias[i], _delta_bias[i]);
                _delta_bias[i] = 0;
                for (size_type j = 0; j < kernels.columns(); ++j) {
                    for (size_type ki = 0; ki < kernel_size.height; ++ki) {
                        for (size_type kj = 0; kj < kernel_size.width; ++kj) {
                            optimizer.updateSingleWeight(kernels(i, j)(ki, kj), _delta_kernels(i, j)(ki, kj));
                            _delta_kernels(i, j)(ki, kj) = 0;
                        }
                    }
                }
            }
        }

        std::ofstream &save(std::ofstream& out) const override {
            using Utils::r_cast_const;

            out.put(CONV2D);

            out.write(r_cast_const(input_size), sizeof(input_size));
            out.write(r_cast_const(kernel_size), sizeof(kernel_size));
            out.write(r_cast_const(bias.n), sizeof(size_type));
            out.write(r_cast_const(padding), sizeof(size_type));

            out.write(r_cast_const(*kernels(0, 0).array), Base::params*sizeof(value_type));
            out.flush();
            return out;
        }

        static Conv2D* load(std::ifstream& in) {
            using Utils::r_cast;

            Shape3D input_size;
            Shape2D kernel_size;
            size_type planes_out = 0;
            size_type padding = 0;

            in.read(r_cast(input_size), sizeof(input_size));
            in.read(r_cast(kernel_size), sizeof(kernel_size));
            in.read(r_cast(planes_out), sizeof(size_type));
            in.read(r_cast(padding), sizeof(size_type));

            auto params = planes_out*input_size.depth*kernel_size.area()+planes_out;
            auto weights = new value_type[params];
            in.read(r_cast(*weights), params*sizeof(value_type));

            return new Conv2D(input_size, kernel_size, planes_out, padding, weights);
        }

        void printHardCodeInitializer(std::ostream &out) const override {
            out << "new Conv2D<T>("<<input_size<<", "<<kernel_size<<", "<<kernels.rows()<<", "<<padding<<", ";
            Utils::printHardCodeArray(kernels(0, 0).array, Base::params, out);
            out << ')';
        }
    };

    template <typename T>
    class MaxPool2D : public Layer<T> {
    public:
        typedef T value_type;
        typedef Layer<value_type> Base;

        typedef typename Base::Data Data;
        typedef typename Base::Matrix Matrix;
    protected:
        const Shape3D input_size;
        const Shape2D kernel_size;

    public:
        constexpr MaxPool2D(Shape3D inputSize, Shape2D kernelSize):
                Base({
                             Utils::ceilDivide(inputSize.height, kernelSize.height),
                             Utils::ceilDivide(inputSize.width, kernelSize.width),
                             inputSize.depth
                     }, 0, "MaxPool2D"), input_size(inputSize), kernel_size(kernelSize) {
            //std::cout << "loaded max pool\n";
            //std::cout << kernel_size.height << ' ' << kernel_size.width << '\n';
            //std::cout << input_size.height << ' ' << input_size.width << ' ' << input_size.depth << '\n';
        }

        void forward(Data _in, Data _out) const override {
            size_type
                    volumeIn  = input_size.height  * input_size.width,
                    volumeOut = Base::output_size.height * Base::output_size.width;
            std::fill(_out, _out + Base::output_size.volume(), -std::numeric_limits<value_type>::max());
            for (size_type plane = 0; plane < input_size.depth; ++plane) {
                Matrix in(_in + volumeIn * plane, input_size.height, input_size.width);
                Matrix out(_out + volumeOut * plane, Base::output_size.height, Base::output_size.width);
                for (size_type i = 0; i < in.rows(); ++i) {
                    size_type iOut = i/kernel_size.height;
                    for (size_type j = 0; j < in.columns(); ++j) {
                        size_type jOut = j/kernel_size.width;
                        if (in(i, j) > out(iOut, jOut)) out(iOut, jOut) = in(i, j);
                    }
                }
                in.array = out.array = nullptr;
            }
        }

        void backward(Data _in, Data _out, Data _deltaIn, Data _deltaOut) const override {
            size_type
                    volumeIn  = input_size.height  * input_size.width,
                    volumeOut = Base::output_size.height * Base::output_size.width;
            for (size_type plane = 0; plane < input_size.depth; ++plane) {
                Matrix in(_in + volumeIn * plane, input_size.height, input_size.width);
                Matrix deltaOut(_deltaOut + volumeIn * plane, input_size.height, input_size.width);

                Matrix out(_out + volumeOut * plane, Base::output_size.height, Base::output_size.width);
                Matrix deltaIn(_deltaIn + volumeOut * plane, Base::output_size.height, Base::output_size.width);

                for (size_type i = 0; i < deltaIn.rows(); ++i) {
                    size_type iIn = i*kernel_size.height;
                    for (size_type j = 0; j < deltaIn.columns(); ++j) {
                        size_type jIn = j*kernel_size.width;
                        auto dIn = deltaIn(i, j);
                        for (size_type ki = 0; ki < kernel_size.height; ++ki) {
                            for (size_type kj = 0; kj < kernel_size.width; ++kj) {
                                deltaOut(iIn + ki, jIn + kj) = (out(i, j) == in(iIn + ki, jIn + kj)? dIn: 0);
                            }
                        }
                    }
                }
                in.array = out.array = deltaIn.array = deltaOut.array = nullptr;
            }
        }

        void updateParameters(Optimizers::Optimizer<value_type> &optimizer) override {
        }

        std::ofstream &save(std::ofstream& out) const override {
            using Utils::r_cast_const;

            out.put(MAXPOOL2D);

            out.write(r_cast_const(input_size), sizeof(input_size));
            out.write(r_cast_const(kernel_size), sizeof(kernel_size));

            return out;
        }

        constexpr static MaxPool2D* load(std::ifstream& in) {
            using Utils::r_cast;

            Shape3D inputSize{};
            Shape2D kernelSize{};

            in.read(r_cast(inputSize), sizeof(inputSize));
            in.read(r_cast(kernelSize), sizeof(kernelSize));

            return new MaxPool2D(inputSize, kernelSize);
        }

        void printHardCodeInitializer(std::ostream &out) const override {
            out << "new MaxPool2D<T>("<<input_size<<", "<<kernel_size<<")";
        }
    };

    template <typename T>
    class ActivationLayer : public Layer<T> {
    public:
        typedef T value_type;
        typedef Layer<value_type> Base;

        typedef typename Base::Data Data;
    protected:
        const size_type size;
        const Activation::ActivationType type;
        const std::pair<std::function<void(Data, Data, size_type)>, std::function<void(Data, Data, Data, size_type)>> functions;

    public:

        constexpr ActivationLayer(Shape3D input, Activation::ActivationType type):
                Base(input, 0, Activation::getActivationName(type)),
                size(input.volume()),
                type(type),
                functions(std::move(Activation::getActivation<value_type>(type))) {
            //std::cout << "loaded ActivationLayer\n";
            //std::cout << input.height << ' ' << input.width << ' ' << input.depth << '\n';
            //std::cout << type << '\n';
        }

        void forward(Data _in, Data _out) const override {
            functions.first(_in, _out, size);
        }

        void backward(Data, Data _out, Data _deltaIn, Data _deltaOut) const override {
            functions.second(_out, _deltaIn, _deltaOut, size);
        }

        void updateParameters(Optimizers::Optimizer<value_type> &optimizer) override {
        }

        std::ofstream &save(std::ofstream& out) const override {
            using Utils::r_cast_const;

            out.put(ACTIVATION);

            out.write(r_cast_const(Base::output_size), sizeof(Base::output_size));
            out.put((char)type);

            return out;
        }

        constexpr static ActivationLayer* load(std::ifstream& in) {
            using Utils::r_cast;

            Shape3D input{};
            char type{};

            in.read(r_cast(input), sizeof(input));
            in.read(r_cast(type), sizeof(type));

            return new ActivationLayer(input, static_cast<Activation::ActivationType>(type));
        }

        void printHardCodeInitializer(std::ostream &out) const override {
            out << "new ActivationLayer<T>("<<Base::output_size<<", static_cast<Activation::ActivationType>("<<type<<"))";
        }
    };

    struct LayerInit {
        LayerType type;
        size_type size[6];
        uint flags = 0;

        template <typename T>
        Layer<T>* create(Shape3D inputSize, std::mt19937 &gen) const {
            switch (type) {
                case DENSE:
                    return new DenseLayer<T>(inputSize.volume(), size[0], gen);
                case CONV2D:
                    return new Conv2D<T>
                            (
                                    inputSize,
                                    {size[0], size[1]},
                                    size[2],
                                    gen,
                                    size[3]
                            );
                case MAXPOOL2D:
                    return new MaxPool2D<T>(inputSize, {size[0], size[1]});
                case ACTIVATION:
                    return new ActivationLayer<T>(inputSize, static_cast<Activation::ActivationType>(flags));
            }
            return nullptr;
        }

        template <typename T>
        static Layer<T>* open(std::ifstream &file) {
            LayerType type{};
            file.read(Utils::r_cast(type), sizeof(type));
            switch (type) {
                case DENSE:
                    return DenseLayer<T>::load(file);
                case CONV2D:
                    return Conv2D<T>::load(file);
                case MAXPOOL2D:
                    return MaxPool2D<T>::load(file);
                case ACTIVATION:
                    return ActivationLayer<T>::load(file);
            }
            return nullptr;
        }
    };

    namespace Loss {
        struct MSE {
            constexpr static const uint gradLayerSkip = 0;

            template <typename T>
            static double eval(T* predictions, T* labels, size_type n) {
                double result = 0.0;
                for (size_type i = 0; i < n; ++i) {
                    auto d = predictions[i] - labels[i];
                    result += d*d;
                }
                return .5 * result;
            }

            template <typename T>
            static void derivative(T* predictions, T* labels, size_type n, T* result) {
                for (size_type i = 0; i < n; ++i) {
                    result[i] = (predictions[i] - labels[i]);
                }
            }
        };

        struct MSETanh {
            constexpr static const size_type gradLayerSkip = 1;

            template <typename T>
            static double eval(T* predictions, T* labels, size_type n) {
                return MSE::eval(predictions, labels, n);
            }

            template <typename T>
            static void derivative(T* predictions, T* labels, size_type n, T* result) {
                for (size_type i = 0; i < n; ++i) {
                    result[i] = (predictions[i] - labels[i]) * (1.0 - predictions[i]*predictions[i]);
                }
            }
        };

        struct CrossEntropySoftmax {
            constexpr static const size_type gradLayerSkip = 1;

            template <typename T>
            static double eval(T* predictions, T* labels, size_type n) {
                double result = 0.0;
                for (size_type i = 0; i < n; ++i) {
                    //std::cout << labels[i] << ' ' << predictions[i] << '\n';
                    result += labels[i]*std::log(predictions[i] + 1e-7);
                }
                //std::cout << "\n\n";
                return -result;
            }

            template <typename T>
            static void derivative(T* predictions, T* labels, size_type n, T* result) {
                //auto labelSum = std::accumulate<T*, T>(labels, labels+n, 0);
                for (size_type i = 0; i < n; ++i) {
                    result[i] = predictions[i] - labels[i];
                }
            }
        };

        template <typename T>
        struct Loss {
            mutable size_type gradLayerSkip=0;
            mutable std::function<double(T*, T*, size_type)> eval;
            mutable std::function<void(T*, T*, size_type, T*)> derivative;

            constexpr Loss() = default;
            constexpr Loss(const Loss &) = default;
            constexpr Loss &operator=(const Loss &) = default;

            template <typename LOSS>
            static Loss create() {
                return {LOSS::gradLayerSkip, LOSS::template eval<T>, LOSS::template derivative<T>};
            }
        };
    }

    template <typename T>
    class LayerBlock {
    public:
        typedef T value_type;
        typedef value_type* Data;

        const Shape3D input_size;
        const size_type size, output_blocks;
        LayerBlock** next;
        Data* buffers{}, * full_cache{};
    protected:
        size_type buffer_size = 0, full_cache_size = 0, params = 0;
        Layer<value_type>** layers;
    public:
        static void getLastOutputBlocks(LayerBlock* root, LayerBlock** output_blocks) {
            size_type i = 0;
            std::vector<LayerBlock<value_type>*> stack{root};
            while (!stack.empty()) {
                LayerBlock<value_type>* back = stack.back();
                stack.pop_back();
                if (back->output_blocks) {
                    for (size_type j = back->output_blocks-1; j < back->output_blocks; --j) stack.push_back(back->next[j]);
                } else output_blocks[i++] = back;
            }
        }

        LayerBlock(Shape3D input_size, size_type size, size_type output_blocks, Layer<value_type>** layers):
                input_size(input_size),
                size(size),
                output_blocks(output_blocks),
                next(new LayerBlock*[output_blocks]),
                buffer_size(input_size.volume()),
                layers(layers)
        {
            for (size_type i = 0; i < size; ++i) {
                params += layers[i]->params;
                buffer_size = std::max(buffer_size, layers[i]->output_size.volume());
                full_cache_size += layers[i]->output_size.volume();
            }

            buffers = createBuffers();
        }

        void printHardCodeInitializer(std::ostream &out, const std::string &varName) {
            out << "auto " << varName << " = new LayerBlock<T>("<<input_size<<", "<<size<<", "<<output_blocks<<", new Layer<T>*["<<size<<"]{\n";
            for (size_type i = 0; i < size; ++i) {
                out << "    ";
                layers[i]->printHardCodeInitializer(out);
                out << ",\n";
            }
            out << "});\n";
            for (size_type output = 0; output < output_blocks; ++output) {
                next[output]->printHardCodeInitializer(out, varName + "->next[" + std::to_string(output) + "]");
            }
        }

        static LayerBlock* createLayerBlock(
                const Shape3D input_size,
                std::initializer_list<LayerInit> init,
                std::mt19937 &gen,
                size_type output_blocks=0)
        {
            auto output_size = input_size;
            auto layers = new Layer<value_type>*[init.size()];
            size_type i = 0;
            for (auto l : init) {
                layers[i] = l.create<T>(output_size, gen);
                output_size = layers[i]->output_size;

                ++i;
            }

            return new LayerBlock(input_size, init.size(), output_blocks, layers);
        }

        static LayerBlock* load(Shape3D input_size, std::ifstream &in) {
            size_type output_blocks, size;

            in.read(Utils::r_cast(output_blocks), sizeof(output_blocks));
            in.read(Utils::r_cast(size), sizeof(size));

            auto layers = new Layer<value_type>*[size];
            for (uint i = 0; i < size; ++i) layers[i] = LayerInit::open<value_type>(in);

            auto r = new LayerBlock(input_size, size, output_blocks, layers);
            for (size_type i = 0; i < output_blocks; ++i) r->next[i] = load(layers[size-1]->output_size, in);
            return r;
        }

        void save(std::ofstream &out) const {
            out.write(Utils::r_cast_const(output_blocks), sizeof(output_blocks));
            out.write(Utils::r_cast_const(size), sizeof(size));
            for (size_type i = 0; i < size; ++i) layers[i]->save(out);
            for (size_type i = 0; i < output_blocks; ++i) next[i]->save(out);
        }

        LayerBlock(const LayerBlock&) = delete;

        LayerBlock(const LayerBlock &o, bool fullCache):
                input_size(o.input_size),
                size(o.size),
                output_blocks(o.output_blocks),
                next(new LayerBlock*[output_blocks]),
                buffer_size(o.buffer_size),
                full_cache_size(o.full_cache_size),
                params(o.params),
                layers(new Layer<value_type>*[size])
        {
            std::copy(o.layers, o.layers + size, layers);

            for (size_type i = 0; i < output_blocks; ++i) next[i] = new LayerBlock(*o.next[i], fullCache);

            buffers = createBuffers();
            if (fullCache) full_cache = createFullCache();
        }

        ~LayerBlock() {
            freeBuffers();
            freeFullCache();
            delete[] layers;
            for (size_type i = 0; i < output_blocks; ++i) delete next[i];
            delete[] next;
        }

        nd_c Data* createBuffers() const {
            auto memory = new value_type[buffer_size * 2];
            return new Data[2]{memory, memory + buffer_size};
        }

        nd_c Data* createFullCache() const {
            auto mem = new value_type[full_cache_size];
            auto r = new Data[size + 1];
            r[0] = nullptr;
            size_type p = 0;
            for (uint i = 0; i < size; ++i) {
                r[i+1] = mem + p;
                p += layers[i]->output_size.volume();
            }
            return r;
        }

        void freeLayers() const {
            for (size_type i = 0; i < size; ++i) delete layers[i];
            for (size_type i = 0; i < output_blocks; ++i) next[i]->freeLayers();
        }

        void freeBuffers() const {
            delete[] *buffers;
            delete[] buffers;
        }

        void freeFullCache() const {
            if (full_cache) {
                delete[] full_cache[1];
                delete[] full_cache;
            }
        }

        nd_c Data getForwardOutput() const {
            return buffers[size&1u];
        }

        nd_c Shape3D getOutputSize() const {
            return layers[size-1]->output_size;
        }

        nd_c size_type getAmountLastOutputBlocks() const {
            if (!output_blocks) return 1;
            size_type a = 0;
            for (size_type i = 0; i < output_blocks; ++i) a += next[i]->getAmountLastOutputBlocks();
            return a;
        }

        nd_c size_type getTotalParameters() const {
            size_type a = params;
            for (size_type i = 0; i < output_blocks; ++i) a += next[i]->getTotalParameters();
            return a;
        }

        void forward(Data in) const {
            layers[0]->forward(in, buffers[1]);
            size_type i = 1;
            for (; i < size; ++i) {
                layers[i]->forward(buffers[i&1u], buffers[!(i&1u)]);
            }
            Data out = getForwardOutput();
            for (size_type j = 0; j < output_blocks; ++j) {
                next[j]->forward(out);
            }
        }

        void forwardCached(Data in) const {
            full_cache[0] = in;
            size_type i = 0;
            for (; i < size; ++i) {
                auto layer = layers[i];
                layer->forward(full_cache[i], full_cache[i + 1]);
            }
            Data out = full_cache[size];
            for (size_type j = 0; j < output_blocks; ++j) {
                next[j]->forwardCached(out);
            }
        }

        double singleBackward(Data labels, const Loss::Loss<value_type> &loss) const {
            auto eval = loss.eval(full_cache[size], labels, layers[size - 1]->output_size.volume());

            size_type i = size-1-loss.gradLayerSkip;
            loss.derivative(full_cache[size], labels, layers[size - 1]->output_size.volume(), buffers[i & 1u]);
            for (; i < size; --i) {
                auto layer = layers[i];
                layer->backward(full_cache[i], full_cache[i + 1], buffers[i & 1u], buffers[!(i & 1u)]);
            }
            return eval;
        }

        Data backwardRecursive() const {
            if (!output_blocks) return buffers[1];
            const size_type outputSize = getOutputSize().volume();
            size_type i = size-1;
            std::fill(buffers[i & 1u], buffers[i & 1u] + outputSize, 0);
            for (size_type j = 0; j < output_blocks; ++j) {
                Data deltaIn = next[j]->backwardRecursive();
                for (size_type k = 0; k < outputSize; ++k) {
                    buffers[i & 1u][k] += deltaIn[k];
                }
            }
            for (; i < size; --i) {
                auto layer = layers[i];
                layer->backward(full_cache[i], full_cache[i + 1], buffers[i & 1u], buffers[!(i & 1u)]);
            }
            return buffers[1];
        }

        void updateParameters(Optimizers::Optimizer<value_type> &optimizer) {
            for (size_type i = 0; i < size; ++i) layers[i]->updateParameters(optimizer);
            for (size_type j = 0; j < output_blocks; ++j) next[j]->updateParameters(optimizer);
        }

        void print(std::ostream &out, int width) const {
            for (size_type i = 0; i < size; ++i) {
                Layer<value_type>* layer = layers[i];
                out << std::left
                    << std::setw(width) << layer->name
                    << std::setw(width) << layer->output_size
                    << std::setw(width) << layer->params
                    << std::endl;
                if (i!=size-1) out << std::string(width*3, '_') << std::endl;
            }
            out << std::string(width*3, '=') << std::endl;
            if (!output_blocks) out << " OUTPUT" << std::endl;
            else out << " CONNECTED BLOCKS:" << std::endl;
            for (size_type j = 0; j < output_blocks; ++j) next[j]->print(out, width);
        }
    };

    template <typename T>
    class Network {
    public:
        typedef T value_type;

        typedef value_type* Data;
        typedef DynamicVector<value_type> Vector;
        typedef LayerBlock<value_type> Layers;

    public:
        const Shape3D input_size;
        size_type total_params, outputs;
        Optimizers::Optimizer<value_type>* optimizer = nullptr;

        Layers* input_block, ** output_blocks;

    public:
        explicit Network(Layers* block):
                input_size(block->input_size),
                total_params(block->getTotalParameters()),
                outputs(block->getAmountLastOutputBlocks()),
                input_block(block),
                output_blocks(new Layers*[outputs])
        {
            Layers::getLastOutputBlocks(input_block, output_blocks);
        }

        ~Network() {
            input_block->freeLayers();
            delete input_block;
            delete optimizer;
        }

        nd_c size_type getTotalParams() const {
            return total_params;
        }

        void setOptimizer(Optimizers::Optimizer<value_type>* newOptimizer) {
            delete optimizer;
            optimizer = newOptimizer;
        }

        DynamicVector<Vector> forward(Data in, Layers* root=nullptr, Layers** outputBlocks=nullptr) const {
            if (!root) {
                root = input_block;
                outputBlocks = output_blocks;
            }
            root->forward(in);
            DynamicVector<Vector> r(outputs);
            for (size_type i = 0; i < outputs; ++i) {
                auto block = outputBlocks[i];
                auto out = block->getForwardOutput();
                r[i] = Vector(out, out+block->getOutputSize().volume());
            }
            return r;
        }

        DynamicVector<double> forwardBackward(Data in, Data* labels, const DynamicVector<Loss::Loss<value_type>> &loss, Layers* root, Layers** outputBlocks) {
            DynamicVector<double> r(outputs);
            root->forwardCached(in);

            for (size_type i = 0; i < outputs; ++i) {
                r[i] = outputBlocks[i]->singleBackward(labels[i], loss[i]);
            }
            root->backwardRecursive();

            return r;
        }

        template<typename SET>
        void train(SET &dataSet, uint epochs, uint miniBatchSizePerThread, const DynamicVector<Loss::Loss<value_type>> &loss, uint numThreads=1) {
            assert(optimizer != nullptr);
            const size_type miniBatchSize = miniBatchSizePerThread*numThreads, batchesPerEpoch = Utils::ceilDivide(dataSet.size, miniBatchSize);

#ifdef USE_THREADS
            auto threads = new std::thread[numThreads];
#endif
            auto blocksIn = new Layers*[numThreads]{};
            auto blocksOut = new Layers**[numThreads]{};

            DynamicVector<double> trainLoss(outputs);

            for (size_type i = 0; i < numThreads; ++i) {
                blocksIn[i] = new Layers(*input_block, true);
                blocksOut[i] = new Layers*[outputs];
                Layers::getLastOutputBlocks(blocksIn[i], blocksOut[i]);
            }

            auto dev = std::random_device();
            std::mt19937 gen(dev());
            auto sequence = new uint[dataSet.size];
            std::iota(sequence, sequence + dataSet.size, 0);

            auto f = [this, &dataSet, &miniBatchSizePerThread, &sequence, &trainLoss, &loss, &blocksIn, &blocksOut](uint thread, uint begin) {
                size_type end = std::min(begin+miniBatchSizePerThread, dataSet.size);
                for (size_type i = begin; i < end; ++i) {
                    size_type imageID = sequence[i];
                    auto input = dataSet.getInput(imageID);
                    auto labels = dataSet.getLabel(imageID);
                    auto l = forwardBackward(input, labels.array, loss, blocksIn[thread], blocksOut[thread]);
                    addReference(trainLoss, l, trainLoss);
                }
            };

            for (size_type epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(sequence, sequence+dataSet.size, gen);
                fill(trainLoss, 0);

                for (uint miniBatch = 0; miniBatch < batchesPerEpoch; ++miniBatch) {
                    for (uint i = 0; i < numThreads; ++i) {
#ifdef USE_THREADS
                        threads[i] = std::thread(f, i, miniBatch*miniBatchSize + i*miniBatchSizePerThread);
#else
                        f(i, miniBatch*miniBatchSize + i*miniBatchSizePerThread);
#endif
                    }
#ifdef USE_THREADS
                    for (uint i = 0; i < numThreads; ++i)
                        threads[i].join();
#endif
                    //std::cerr << "mini batch no. " << miniBatch << " done!\n";
                    updateParameters();
                }

                std::cerr << "Epoch: " << epoch+1 << ", average train loss: [";
                for (size_type i = 0; i < outputs; ++i) {
                    std::cerr << trainLoss[i] / (double)dataSet.size;
                    if (i+1 != outputs) std::cerr << ", ";
                }
                std::cerr << "]" << std::endl;
                optimizer->postEpoch();
            }

            for (size_type i = 0; i < numThreads; ++i) {
                delete blocksIn[i];
            }
#ifdef USE_THREADS
            delete[] threads;
#endif
            delete[] blocksIn;
            delete[] blocksOut;
            delete[] sequence;
        }

        void updateParameters() {
            optimizer->preUpdate();
            input_block->updateParameters(*optimizer);
        }

        void print(std::ostream &out=std::cout) {
            const int width = 16;
            out << "Input Size: " << input_size << "\n";
            out << "Output Blocks: " << outputs << "\n";
            out << "Total Params: " << total_params << "\n\n";
            out << std::left << std::setw(width) << "Layer" << std::setw(width) << "Output Size" << std::setw(width) << "Parameters" << std::endl;
            out << std::string(width*3, '=') << std::endl;
            input_block->print(out, width);
        }

        void printHardCodeInitializer(std::ostream &out, const std::string &varName="network") {
            auto root = varName + "Root";
            input_block->printHardCodeInitializer(out, root);
            out << "auto " << varName << " = new Network<T>(" << root << ");\n";
        }

        void save(std::ofstream &out) const {
            out.write(Utils::r_cast_const(input_size), sizeof(input_size));
            input_block->save(out);
        }

        static Network<value_type>* load(std::ifstream &in) {
            Shape3D input_size;

            in.read(Utils::r_cast(input_size), sizeof(input_size));

            return new Network<value_type>(Layers::load(input_size, in));
        }
    };
}

namespace Game {
    using Utils::TwoBitArray;
    using Utils::BitSet64;

    constexpr const static uint WIDTH = 7;
    constexpr const static uint HEIGHT = 9;
    constexpr const static uint AREA = WIDTH*HEIGHT;

    enum Direction: uint8_t {
        UP,
        DOWN,
        LEFT,
        RIGHT
    };

    nd_c Direction operator!(Direction dir) {
        switch (dir) {
            case UP: return DOWN;
            case DOWN: return UP;
            case RIGHT: return LEFT;
            case LEFT: return RIGHT;
            default: return dir;
        }
    }

    static const constexpr int TRANSLATIONS[] {
    //  UP DOWN LEFT RIGHT
        -(int)WIDTH,
         WIDTH,
        -(int)1,
         1
    };

    struct Position {
        uint pos = -1;

        constexpr Position(uint x, uint y): pos(y * WIDTH + x) {
        }

        constexpr Position(uint pos): pos(pos) {
        }

        constexpr Position() = default;

        constexpr operator uint() const {
            return pos;
        }

        nd_c uint x() const {
            return pos % WIDTH;
        }

        nd_c uint y() const {
            return pos / WIDTH;
        }

        constexpr Position &translate(Direction d) {
            pos += TRANSLATIONS[d];
            return *this;
        }

        nd_c static Position mirrorHorizontal(Position p) {
            return WIDTH-p.x()-1 + WIDTH*p.y();
        }

        nd_c bool isBorder(Direction dir) const {
            switch (dir) {
                case UP: return pos < WIDTH;
                case DOWN: return pos >= WIDTH * (HEIGHT-1);
                case RIGHT: return x() == WIDTH - 1;
                case LEFT: return !x();
                default: return true;
            }
        }

        explicit operator std:: string() const {
            return std::string{char(y() + 'a'), char(x() + 'a')};
        }
    };

    struct JointPosition {
        uint hash = 255;

        constexpr JointPosition(Position p, bool b) {
            hash = (p << 1) | b;
        }

        constexpr JointPosition(uint hash): hash(hash) {
        }

        constexpr JointPosition() = default;

        nd_c Position pos() const {
            return hash >> 1;
        }

        nd_c bool orientation() const {
            return hash & 1;
        }

        nd_c bool null() const {
            return hash == 255;
        }

        constexpr operator uint() const {
            return hash;
        }
    };

    struct Move {

        uint hash = 255;

        constexpr Move() = default;

        constexpr Move(uint a): hash(a) {
        }

        constexpr Move(Position p, uint t) {
            hash = p | (t << 6);
        }

        constexpr Move(const char* str): Move(Position(str[1] - 'a', str[0] - 'a'), str[2] == 'l'? 3: (str[2] == 's')? 2: 1) {
        }

        explicit operator std:: string() const {
            return (std::string)pos() + char(type() == 3? 'l': type() == 2? 's': 'r');
        }

        friend std::istream &operator>>(std::istream &in, Move &move) {
            std::string c;
            in >> c;
            move.hash = Move(c.c_str()).hash;
            return in;
        }

        nd_c bool operator==(const Move &o) const {
            return hash == o.hash;
        }

        nd_c Position pos() const {
            return hash & 63u;
        }

        nd_c uint type() const {
            return hash >> 6;
        }
    };

    template <typename T>
    class Board {
    public:
        typedef T floating_type;
#ifdef COLOR_BOARD
        //                                   RESET  BLUE       RED         BLUE2      RED2  GRAY (5)
        constexpr static const char* COLOR[] {"39", "38;5;17", "38;5;196", "38;5;31", "91", "38;5;240"};
#endif
    private:
        // Union Find:
        constexpr const static uint AIR  = WIDTH * HEIGHT * 2;
        constexpr const static uint WALL[] {AIR+1, AIR+1, AIR+2, AIR+3};
        nd_c static Direction getWallDirection(uint wallValue) {
            return static_cast<Direction>(wallValue - WALL[0] + 1);
        }
        struct UnionSet {
            uint borders[2]{};
            JointPosition tips[2]{}, root = -1;
            uint size = 1;
#ifdef COLOR_BOARD
            uint8 color = 0;
#endif

            UnionSet() = default;
            constexpr UnionSet(const UnionSet&) = default;
            constexpr UnionSet(UnionSet&&) noexcept = default;

            constexpr UnionSet& operator=(UnionSet&&) noexcept = default;

            nd_c bool getTip(JointPosition tip) const {
                return tips[0] != tip;
            }

            constexpr void handleBorder(uint border) {
                if (!borders[0]) borders[0] = border;
                else borders[1] = border;
            }

            constexpr void reset() {
                size = 1;
                borders[0] = borders[1] = 0;
                tips[0] = tips[1] = root;
            }

            nd_c bool enclosed() const {
                return borders[1];
            }

            constexpr bool operator<(const UnionSet &o) const {
                return size < o.size;
            }
        };

    public:
        struct BoardChange {
            struct DecodeChange {
                const uint plane = 0, mPlane = 0, pos = 0;
                const floating_type x = 0;

                DecodeChange() = delete;

                constexpr DecodeChange(uint plane, uint mPlane, uint pos, floating_type x):
                        plane(plane), mPlane(mPlane), pos(pos), x(x)
                {
                }
            };

            Position move;
            int score = 0;
            vector_stack<std::pair<JointPosition, uint>> union_parent_stack;
            vector_stack<UnionSet> union_unite_stack;
            vector_stack<std::pair<uint, uint>> p_score_stack;
            vector_stack<DecodeChange> decoded_stack;

            BoardChange() = default;
        };
    private:

        nd_c uint findUnion(uint pos, BoardChange* change) {
            if (_union_parents[pos] == pos) return pos;
            uint p = findUnion(_union_parents[pos], change);
            if (p != _union_parents[pos]) {
                if (change) change->union_parent_stack.emplace(pos, _union_parents[pos]);
                _union_parents[pos] = p;
            }
            return p;
        }

        nd_c uint findUnionConst(uint pos) const {
            if (_union_parents[pos] == pos) return pos;
            return findUnionConst(_union_parents[pos]);
        }

        constexpr UnionSet* unite(JointPosition aTip, uint b, JointPosition bTip, BoardChange* change) {
            uint a = findUnion(aTip, change);
            b = findUnion(b, change);
            auto A = &_union_sets[a], B = &_union_sets[b];
            if (a == b) return A;
            bool bTipI = B->getTip(bTip);
            JointPosition tip = A->tips[!A->getTip(aTip)];

            if (change) {
                if (change->union_parent_stack.empty()
                                || change->union_parent_stack.top().first != a)
                    change->union_parent_stack.emplace(a, _union_parents[a]);
                if (a != tip) change->union_parent_stack.emplace(tip, _union_parents[tip]);
                change->union_unite_stack.emplace(*B);
            }

            _union_parents[a] = b;
            _union_parents[tip] = b;
            B->tips[bTipI] = tip;
            B->size += A->size;
            if (A->borders[0]) B->handleBorder(A->borders[0]);
            return B;
        }

        UnionSet _union_sets[WIDTH * HEIGHT * 2ul];
        uint _union_parents[WIDTH * HEIGHT * 2ul]{};

        std::array<uint, WIDTH * HEIGHT> _state_grid{};
        int _score[2]{50, 50};
        uint _potential_score_sum[2] = {HEIGHT, HEIGHT}, _potential_score[2][HEIGHT]{};
        BitSet64 _legal_moves = (1ull << (WIDTH * HEIGHT)) - 1ull;
        bool _turn = false, _game_over = false;
        Move _last_move{};

    public:
        enum Plane : uint {
            FREE,
            TYPE_R,
            TYPE_S,
            TYPE_L,

            CONNECTED_GREY1,
            CONNECTED_GREY2,
            CONNECTED_GREY3,
            CONNECTED_GREY4,

            CONNECTED_LEFT1,
            CONNECTED_LEFT2,
            CONNECTED_LEFT3,
            CONNECTED_LEFT4,

            CONNECTED_RIGHT1,
            CONNECTED_RIGHT2,
            CONNECTED_RIGHT3,
            CONNECTED_RIGHT4,
        };
        constexpr static const uint planes = 16;

        constexpr static uint getConnectPlane(uint border, Direction dir) {
            return CONNECTED_GREY1 + (border-1)*4 + dir;
        }

        struct Decoder {
            typedef floating_type* Data;

            Data normal = new floating_type[planes * AREA]{};
            Data mirror = new floating_type[planes * AREA]{};

            constexpr Decoder() {
                std::fill(normal, normal + AREA, 1);
                std::fill(mirror, mirror + AREA, 1);

                for (uint i = 0; i < HEIGHT; ++i) {
                    setConnected(LEFT, LEFT, {0, i});
                    setConnected(RIGHT, RIGHT,{WIDTH-1, i});
                }

                for (uint i = 0; i < WIDTH; ++i) {
                    setConnected(UP, UP, {i, 0});
                    setConnected(DOWN, DOWN,{i, HEIGHT-1});
                }
            }

            ~Decoder() {
                delete[] normal;
                delete[] mirror;
            }

            nd_c Data getPerspective(bool side) const {
                return side? mirror: normal;
            }

            constexpr void set(uint plane, uint mirroredPlane, Position pos, floating_type x=1, BoardChange* log=nullptr) const {
                if (log) log->decoded_stack.emplace(plane, mirroredPlane, pos, normal[plane * AREA + pos]);
                normal[plane * AREA + pos] = x;
                mirror[mirroredPlane * AREA + Position::mirrorHorizontal(pos)] = x;
            }

            constexpr void setConnected(uint border, Direction dir, Position pos, floating_type x=1, BoardChange* log=nullptr) const {
                uint mDir = dir>=LEFT?! dir:dir;
                if (border >= LEFT) {
                    border -= 2;
                    set(CONNECTED_LEFT1+border*4+dir, CONNECTED_RIGHT1-border*4+mDir, pos, x, log);
                } else {
                    set(CONNECTED_GREY1+dir, CONNECTED_GREY1+mDir, pos, x, log);
                }
            }

            constexpr void setTileState(uint type, Position pos, bool x=true, BoardChange* log=nullptr) const {
                set(type, 4-type, pos, x);
                set(FREE, FREE, pos, !x, log);
            }

            void print(std::ostream &out=std::cout) const {
                for (auto arr : {normal, mirror}) {
                    for (uint plane = 0; plane < planes; ++plane) {
                        if (plane % 4 == 0) out << "\n\n\n";
                        out << plane << ":\n";
                        for (uint y = 0; y < HEIGHT; ++y) {
                            for (uint x = 0; x < WIDTH; ++x) {
                                out << arr[plane * AREA + y * WIDTH + x];
                            }
                            out << '\n';
                        }
                        out << '\n';
                    }
                    out << "\n\n\n";
                }
            }
        };

        const Decoder _decoded{};

    private:

        nd_c JointPosition neighbour(Position p, Direction d) const {
            if (p.isBorder(d)) return WALL[d];
            return getJoint(p.translate(d), !d);
        }

        constexpr void setPotentialScore(bool side, uint y, uint value, BoardChange* log) {
            if (log) log->p_score_stack.emplace(y + side*HEIGHT, _potential_score[side][y]);
            _potential_score_sum[side] += value - _potential_score[side][y];
            _potential_score[side][y] = value;
        }

        nd_c bool getBorderTip(const UnionSet &set, Direction border) const {
            auto pos = set.tips[0].pos();
            if (pos.isBorder(border) && ORIENTATIONS[_state_grid[pos]][border] == set.tips[0].orientation()) return false;
            return true;
        }
    public:

        static const constexpr Direction SIDES[][4] {
            {},
            {RIGHT, LEFT, UP, DOWN},
            {UP, LEFT, DOWN, RIGHT},
            {RIGHT, LEFT, DOWN, UP},
        };

        static const constexpr uint ORIENTATIONS[][4] {
        //  UP DOWN LEFT RIGHT
            {},
            {0, 1, 1, 0},// r
            {0, 0, 1, 1},// s
            {1, 0, 1, 0},// l
        };

        Board() {
            for (uint i = 0; i < AIR; ++i) {
                if (i < HEIGHT) _potential_score[0][i] = _potential_score[1][i] = 1;
                _union_sets[i].root = _union_parents[i] = i;
                _union_sets[i].reset();
            }
        }

        nd_c int getScore(bool s) const {
            return _score[s];
        }

        nd_c uint getPotentialScore(bool s) const {
            return _potential_score_sum[s];
        }

        nd_c BitSet64 getLegalMoves() const {
            return _legal_moves;
        }

        nd_c bool getTurn() const {
            return _turn;
        }

        nd_c const Decoder &getDecodedState() const {
            return _decoded;
        }

        nd_c Move getLastMove() const {
            return _last_move;
        }

        nd_c JointPosition getJoint(Position p, Direction dir) const {
            uint t = _state_grid[p];
            if (!t) return AIR;
            return {p, static_cast<bool>(ORIENTATIONS[t][dir])};
        }

        Move randomMove(Utils::Random &rand=*Utils::RNG) const {
            uint r = rand.nextInt(_legal_moves.count()), i = 0;
            for (auto pos : _legal_moves) {
                if (i++ == r) return {Position(pos), rand.nextBoolean()? 1u: 3u};
            }
            return {0u};
        }

        void play(Move move, BoardChange* log=nullptr) {
            const Position pos = move.pos();
            if (log) {
                log->move = pos;
                log->score = _score[_turn];
            } else _last_move = move;
            const uint j0 = pos << 1, j1 = j0 | 1u;
            const uint type = move.type();
            _decoded.setTileState(type, pos);
            _state_grid[pos] = type;
            _legal_moves.unset(pos);

            {
                uint neighbours[4]{};
                UnionSet* sets[] = {&_union_sets[j0], &_union_sets[j1]};
                uint counts[5]{}; // 2-bit index: bit2 = set, bit1 = side; index 4 is extra count
                uint set;
                for (uint i = 0; i < 4u; ++i) {
                    auto dir = SIDES[type][i];
                    const JointPosition neighbourTip = neighbour(pos, dir);
                    neighbours[i] = neighbourTip;
                    if (neighbours[i] == AIR) continue;
                    if (neighbours[i] < AIR) neighbours[i] = findUnion(neighbours[i], log);
                    set = i & 1u; // 0 1 0 1 (boolean)

                    if (neighbours[i] < AIR) { // has neighbour
                        if (i & 2u) {
                            if (neighbours[i-2] < AIR && neighbours[i] == sets[set]->root) {
                                // cycle
                                _score[_turn] -= 5;
                                #ifdef COLOR_BOARD
                                sets[set]->color = 3 + _turn;
                                #endif
                                continue;
                            }
                        }

                        auto border = _union_sets[neighbours[i]].borders[0]; // border is WALL value
                        if (border) _decoded.setConnected(getWallDirection(border), dir, pos, 0, log);
                        uint count_index = ((i & 1u) << 1) | (border - WALL[2]);
                        if (border > WALL[0]) { // > WALL[0] is blue/red wall
                            if (i & 2u &&
                                    neighbours[(i ^ 1u) & 1u] < AIR &&
                                    neighbours[i] == sets[set ^ 1u]->root) {
                                counts[4] = _union_sets[neighbours[i]].size - counts[count_index ^ 2u] - 1;
                            } else counts[count_index] = _union_sets[neighbours[i]].size;
                        } else if (!set) counts[4] = _union_sets[neighbours[i]].size;

                        auto united = unite(j0 | set, neighbours[i], neighbourTip, log);
                        if (sets[set] == sets[set ^ 1u]) sets[set ^ 1u] = united;
                        sets[set] = united;
                    } else {
                        sets[set]->handleBorder(neighbours[i]);
                        _decoded.setConnected(dir, dir, pos, 0, log);
                    }

                    if (sets[set]->enclosed()) {
                        if (sets[set]->borders[0] != WALL[0]) {
                            if (sets[set]->borders[0] == sets[set]->borders[1]) {
                                // connected color with same color
                                _score[_turn] -= 3;
                                #ifdef COLOR_BOARD
                                sets[set]->color = 3 + _turn;
                                #endif
                            } else if (sets[set]->borders[1] != WALL[0]) {
                                // connected to other side
                                _score[_turn] += static_cast<int>((sets[0] == sets[1]) ?
                                                    (counts[_turn] | counts[_turn + 2]) + counts[4] + 2 : // edge case
                                                    counts[(set << 1) | _turn] + 1 // normal case
                                                 );
                                #ifdef COLOR_BOARD
                                sets[set]->color = _turn + 1;
                            } else sets[set]->color = 5;
                        } else sets[set]->color = 5;
                        #else
                            }
                        }
                        #endif
                    } else if (sets[set]->borders[0] == WALL[0]) {
                        #ifdef COLOR_BOARD
                        sets[set]->color = 5;
                        #endif
                    }
                }
                for (set = 0; set < 2; ++set) {
                    auto s = sets[set];
                    if (set && sets[0] == sets[1]) break;
                    if (s->borders[0]) {
                        if (s->enclosed()) {
                            bool i;
                            if (s->borders[0] != WALL[0]) {
                                auto border = getWallDirection(s->borders[0]);
                                i = getBorderTip(*s, border);
                                setPotentialScore(border == RIGHT, s->tips[i].pos().y(), 0, log);
                            } else if (s->borders[1] != WALL[0])
                                i = !getBorderTip(*s, getWallDirection(s->borders[1]));
                            if (s->borders[1] != WALL[0]) {
                                setPotentialScore(WALL[3] == s->borders[1], s->tips[!i].pos().y(), 0, log);
                            }
                        } else {
                            if (s->borders[0] >= WALL[0]) {
                                auto border = getWallDirection(s->borders[0]);
                                bool i = getBorderTip(*s, border);
                                auto otherTip = s->tips[!i];
                                auto otherTipPos = otherTip.pos();

                                for (uint side = 0; side < 2; ++side) {
                                    auto dir = SIDES[_state_grid[otherTipPos]][side*2+otherTip.orientation()];
                                    if (otherTipPos.isBorder(dir)) continue;
                                    auto p = Position(otherTipPos).translate(dir);
                                    if (!_state_grid[p]) {
                                        _decoded.setConnected(border, !dir, p, 1, log);
                                        //if (!log) std::cerr << p.operator std::string() << " " << !dir << " " << border << " " << getConnectPlane(border, !dir) << '\n';
                                    }
                                }

                                if (s->borders[0] != WALL[0]) {
                                    setPotentialScore(border == RIGHT, s->tips[i].pos().y(), s->size + 1, log);
                                }
                            }
                        }
                    }
                }

                /*
                if (!log) {
                    std::cerr << _potential_score_sum[0] << " " << _potential_score_sum[1] << "\n";
                    for (uint i = 0; i < HEIGHT; ++i)
                        std::cerr << _potential_score[0][i] << ' ' << _potential_score[1][i] << '\n';
                    std::cerr << std::endl;
                }
*/
            }
            _game_over = !_legal_moves;
            _turn = !_turn;

            if (!log) {
                //_decoded.print(std::cerr);
                //print(std::cerr);
            }
        }

        void undo(BoardChange &change) {
            _turn = !_turn;
            _game_over = false;
            while (!change.decoded_stack.empty()) {
                auto &d = change.decoded_stack.top();
                _decoded.set(d.plane, d.mPlane, d.pos, d.x);
                change.decoded_stack.pop();
            }
            _decoded.setTileState(_state_grid[change.move], change.move, false);
            _state_grid[change.move] = 0;
            _legal_moves.orSet(change.move);
            _score[_turn] = change.score;
            while (!change.p_score_stack.empty()) {
                uint i = change.p_score_stack.top().first;
                setPotentialScore(i >= HEIGHT, i % HEIGHT, change.p_score_stack.top().second, nullptr);
                change.p_score_stack.pop();
            }
            while (!change.union_parent_stack.empty()) {
                _union_parents[change.union_parent_stack.top().first] = change.union_parent_stack.top().second;
                change.union_parent_stack.pop();
            }
            while (!change.union_unite_stack.empty()) {
                _union_sets[change.union_unite_stack.top().root] = std::forward<UnionSet>(change.union_unite_stack.top());
                change.union_unite_stack.pop();
            }
            _union_sets[change.move << 1].reset();
            _union_sets[(change.move << 1) | 1u].reset();
        }

        nd_c bool isOver() const {
            return _game_over;
        }

        std::ostream &print(std::ostream &out) const {
            constexpr const uint8 box_width = 5, box_height = 3;
            constexpr const bool grid = false, dots_only = false;
            char r[WIDTH * box_width][HEIGHT * box_height];
            #ifdef COLOR_BOARD
            std::string colors[WIDTH * box_width][HEIGHT * box_height]{};
            constexpr const char* BG = "\033[49m";
            #endif
            for (uint8 y = 0; y < HEIGHT * box_height; ++y) {
                for (auto & x : r) {
                    x[y] = ' ';
                }
            }
            for (uint8 y = 0; y < HEIGHT; ++y) {
                for (uint8 x = 0; x < WIDTH; ++x) {
                    auto type = _state_grid[y * WIDTH + x];
                    if (!type) continue;

                    r[x * box_width][y * box_height + 1] = '-';
                    r[x * box_width + 4][y * box_height + 1] = '-';
                    #ifdef COLOR_BOARD
                    colors[x * box_width][y * box_height + 1] = COLOR[_union_sets[findUnionConst(getJoint(Position(x, y), LEFT))].color];
                    colors[x * box_width + 4][y * box_height + 1] = COLOR[_union_sets[findUnionConst(getJoint(Position(x, y), RIGHT))].color];
                    auto c0 = COLOR[_union_sets[findUnionConst(JointPosition(Position(x, y), false))].color];
                    auto c1 = COLOR[_union_sets[findUnionConst(JointPosition(Position(x, y), true))].color];
                    #endif
                    switch (type) {
                        case 2:
                            r[x * box_width + 2][y * box_height + 1] = '+';

                            r[x * box_width + 2][y * box_height] = '|';
                            r[x * box_width + 2][y * box_height + 2] = '|';
                            r[x * box_width + 3][y * box_height + 1] = '-';
                            r[x * box_width + 1][y * box_height + 1] = '-';
                            #ifdef COLOR_BOARD
                            if (std::strcmp(c0, "0") != 0) colors[x * box_width + 2][y * box_height + 1] += c0;
                            else colors[x * box_width + 2][y * box_height + 1] += c1;
                            colors[x * box_width + 2][y * box_height] = c0;
                            colors[x * box_width + 2][y * box_height + 2] = c0;
                            colors[x * box_width + 3][y * box_height + 1] = c1;
                            colors[x * box_width + 1][y * box_height + 1] = c1;
                            #endif
                            break;
                        case 3:
                            r[x * box_width + 1][y * box_height] = '/';
                            r[x * box_width + 3][y * box_height + 2] = '/';
                            #ifdef COLOR_BOARD
                            colors[x * box_width + 1][y * box_height] = c1;
                            colors[x * box_width + 3][y * box_height + 2] = c0;
                            #endif
                            break;
                        case 1:
                            r[x * box_width + 1][y * box_height + 2] = '\\';
                            r[x * box_width + 3][y * box_height] = '\\';
                            #ifdef COLOR_BOARD
                            colors[x * box_width + 1][y * box_height + 2] = c1;
                            colors[x * box_width + 3][y * box_height] = c0;
                            #endif
                            break;
                        default:
                            continue;
                    }
                }
            }
            #ifdef COLOR_BOARD
            out << BG;
            #endif
            out << "   ";
            for (uint8 x = 0; x < WIDTH; ++x) {
                out << std::string(box_width/2+1, ' ') << char(x+'a') << std::string(box_width/2-!grid, ' ');
            }
            #ifdef COLOR_BOARD
            out << "\033[49m";
            #endif
            out << "\n";
            for (uint8 y = 0;; ++y) {
                #ifdef COLOR_BOARD
                out << BG;
                #endif
                if (y % box_height == box_height/2) out << ' ' << char((y/box_height) + 'a') << ' ';
                else out << "   ";
                if (!(y % box_height) && (grid || y == HEIGHT * box_height || !y)) {
                    for (uint8 x = 0; x < WIDTH; ++x) {
                        if (grid || !x) out << '+';
                        out << std::string(box_width, !(y == HEIGHT * box_height || !y) && dots_only? ' ': '-');
                    }
                    out << '+';
                    #ifdef COLOR_BOARD
                    out << "\033[49m";
                    #endif
                    out << '\n';
                    #ifdef COLOR_BOARD
                    out << BG;
                    #endif
                    out << "   ";
                }
                if (y == HEIGHT * box_height) break;
                for (uint8 x = 0; x < WIDTH * box_width; ++x) {
                    if (!(x % box_width) && (grid || !x)){
                        #ifdef COLOR_BOARD
                        if (!x) out << std::string() + "\033[" + COLOR[1] + "m|\033[39m";
                        else
                        #endif
                        out << (!(x == WIDTH * box_width || !x) && dots_only? ' ': '|');
                    }
                    #ifdef COLOR_BOARD
                    out << std::string() + "\033[" + colors[x][y] + "m";
                    #endif
                    out << r[x][y];
                    #ifdef COLOR_BOARD
                    out << "\033[39m";
                    #endif
                }
                #ifdef COLOR_BOARD
                out << std::string() + "\033[" + COLOR[2] + "m|\033[0m";
                #else
                out << '|';
                #endif
                out << '\n';
            }
            std::string score[] = {std::to_string(_score[0]), std::to_string(_score[1])};
            return out << score[0] <<
                       std::string((box_width + grid) * WIDTH - score[0].length() - score[1].length() + 1 + !grid,
                                   ' ') << score[1] << '\n';
        }

    };
}

namespace MoveFinder {
    constexpr const static uint input_size = Game::AREA * Game::Board<float>::planes, output_size = Game::AREA * 3;

    template <typename T>
    struct Experience {
        typedef T* Data;

        T score = 0;
        Data
                input = new T[input_size],
                prior_label = new T[output_size]{};

        ~Experience() {
            delete[] input;
            delete[] prior_label;
        }
    };

    template <typename T>
    struct Collector {
        std::vector<Experience<T>*> vec{};

        constexpr void complete(int finalScore) {
            for (auto e : vec) (e->score = finalScore - e->score) /= 40.;
        }
    };

    template <typename T>
    struct DataSet {
        typedef T* Data;

        std::vector<Experience<T>*> e;
        uint size = 0;

        ~DataSet() {
            for (auto i : e) delete i;
        }

        nd_c Data getInput(uint i) const {
            return e[i]->input;
        }

        nd_c Utils::Math::Matrix::DynamicVector<Data> getLabel(uint i) const {
            return {e[i]->prior_label, &e[i]->score};
        }

        void add(const std::vector<Experience<T>*> &v) {
            std::copy(v.cbegin(), v.cend(), std::back_inserter(e));
        }
    };


    template <typename T>
    class MoveController {
    public:
        const bool side;
        Game::Board<T>* board;
    protected:
        explicit MoveController(bool side): side(side) {}
    public:
        virtual Game::Move suggest() = 0;
        virtual ~MoveController() = default;
    };

    template <typename T>
    class RandomBot : public MoveController<T> {
    public:
        typedef MoveController<T> Base;

        RandomBot(bool side): Base(side) {}

        Game::Move suggest() override {
            return Base::board->randomMove();
        }
    };

    namespace BoardEvaluation {
        struct PotentialScore {
            int score[2];
            uint p_score[2];

            constexpr explicit PotentialScore(bool maximizing):
                    score{maximizing ? INT32_MIN : INT32_MAX, maximizing ? INT32_MAX : INT32_MIN},
                    p_score{maximizing ? 0 : UINT32_MAX, maximizing ? 0 : UINT32_MAX}
                    {}

            template<typename T>
            PotentialScore(const Game::Board<T> &b, bool side):
                    score{b.getScore(side), b.getScore(!side)},
                    p_score{b.getPotentialScore(side), b.getPotentialScore(!side)} {}

            PotentialScore() = default;
            PotentialScore(const PotentialScore&) = default;

            nd_c bool operator>=(const PotentialScore &o) const {
                return
                    score[0] > o.score[0] ||
                    (score[0] == o.score[0] && (p_score[0] > o.p_score[0] ||
                    (p_score[0] == o.p_score[0] && (p_score[1] > o.p_score[1] ||
                    (p_score[1] == o.p_score[1] && score[1] <= o.score[1])))));
            }

            nd_c bool operator<(const PotentialScore &o) const {
                return !operator>=(o);
            }
        };
    }

    constexpr static Utils::BitSet64 slice1 = 0x102040810204081, slice2 = 0x4081020408102040;

    uint m_count = 0;
    template <typename T, typename S>
    T minimax(
            Game::Board<S> &board,
            bool side,
            uint depth,
            bool maximizing=true,
            T alpha=T{false}, T beta=T{true}
            ) {
        typedef T Evaluation;
        ++m_count;
        if (!depth || board.isOver()) {
            return {board, side};
        }
        Evaluation best(maximizing);
        typename Game::Board<S>::BoardChange undo;
        uint i = 0;
        Utils::BitSet64 currentSlice = side? slice2: slice1;
        while (i < Game::WIDTH)  {
            for (auto pos : board.getLegalMoves() & currentSlice) {
                for (uint t = 1; t < 4; ++t) {
                    board.play({pos, t}, &undo);
                    Evaluation next = minimax(board, side, depth-1, !maximizing, alpha, beta);
                    board.undo(undo);

                    if (maximizing) {
                        if (best < next) {
                            best = next;
                            if (best >= beta) return best;
                            if (best < alpha) alpha = best;
                        }
                    } else {
                        if (next < best) {
                            best = next;
                            if (alpha >= best) return best;
                            if (beta < best) beta = best;
                        }
                    }
                }
            }

            if (side) currentSlice >>= 1;
            else currentSlice <<= 1;
            ++i;
        }
        return best;
    }

    template <typename EVALUATOR, typename T>
    class MinimaxMoveController: public MoveController<T> {
    public:
        typedef MoveController<T> Base;
        typedef EVALUATOR Evaluation;

        uint depth = 3;

        Collector<T>* collector;

        MinimaxMoveController(bool side, Collector<T>* collector=nullptr): Base(side), collector(collector) {}

        Game::Move suggest() override {
            uint count = Base::board->getLegalMoves().count();
            /*
            if (count <= 25) depth = 4;
            if (count <= 13) depth = 5;
            if (count <= 10) depth = 6;
            if (count <= 9) depth = 7;
            */
            Evaluation bestScore(true), alpha(false);
            Game::Move moves[Game::WIDTH * Game::HEIGHT * 3];
            uint size = 0;
            typename Game::Board<T>::BoardChange undo{};
            //std::cerr << "WORD: " << board.getLegalMoves().word << '\n';
            uint i = 0;
            Utils::BitSet64 currentSlice = Base::side? slice2: slice1;
            while (i < Game::WIDTH) {
                //std::cerr << std::bitset<64>(currentSlice.word) << '\n';
                for (auto pos : Base::board->getLegalMoves() & currentSlice) {
                    //std::cerr << (std::string)Game::Position(pos) << ',';
                    for (uint t = 1; t < 4; ++t) {
                        Game::Move m(pos, t);
                        //std::cerr << (std::string)m << '\n';
                        Base::board->play(m, &undo);
                        auto next = minimax<Evaluation>(*Base::board, Base::side, depth-1, false, alpha);
                        Base::board->undo(undo);

                        if (next >= bestScore) {
                            if (bestScore < next) {
                                bestScore = next;
                                size = 0;
                            }
                            moves[size++] = m;
                            if (bestScore < alpha) alpha = bestScore;
                        }
                    }
                }
                if (Base::side) currentSlice >>= 1;
                else currentSlice <<= 1;
                ++i;
            }
            //std::cerr << count << ": board states: " << m_count << " (depth: " << depth << ')' << '\n';
            m_count = 0;
            uint r = Utils::RNG->nextInt(size);

            if (collector) {
                auto e = new Experience<T>{(T)Base::board->getScore(Base::side)};
                auto in = Base::board->getDecodedState().getPerspective(Base::side);
                auto p = 1./(T)size;
                std::copy(in, in + input_size, e->input);
                for (uint j = 0; j < size; ++j) {
                    auto move = moves[j];
                    auto pos = Base::side? Game::Position::mirrorHorizontal(move.pos()): move.pos();
                    uint t = Base::side? (3-move.type()): (move.type()-1);
                    e->prior_label[pos*3+t] = p;
                }
                collector->vec.push_back(e);
            }
            /*
            std::cerr << '\n';
            for (uint j = 0; j < size; ++j) {
                std::cerr << j << ":" << (std::string)moves[j] << ',';
            }
            std::cerr << '\n';
            std::cerr << i << ' ' << size << '\n';
             */
            return moves[r];
        }
    };

    template <typename T>
    class NeuralNetworkTreeSearch: public MoveController<T> {
    public:
        typedef MoveController<T> Base;
        typedef T floating_type;
        typedef floating_type* Data;

        typedef Game::Board<floating_type> Board;
        typedef Game::Move Move;

        template <typename S>
        using Vector = Utils::Math::Matrix::DynamicVector<S>;

        typedef Vector<floating_type> FloatVector;
        typedef DeepLearning::Network<floating_type> Network;
        typedef DeepLearning::LayerBlock<floating_type> LayerBlock;

        constexpr const static floating_type
                explorationRate = 3,
                dirichletAlpha = 0.03,
                weightPrior = 5,
                weightDirichlet = 1;

        struct TreeNode {
            uint visits = 0, not_fully_visited;
            int total_score_gain = 0, score_gain = 0;
            floating_type value = 0, prior;
            Move move;

            TreeNode* parent = nullptr;
            std::vector<TreeNode*> children{};

            constexpr explicit TreeNode(uint legalMoves):
                    not_fully_visited(legalMoves*3), prior(0), move{}, parent(nullptr) {}

            constexpr TreeNode(TreeNode &parent, Move move, floating_type prior):
                    not_fully_visited(parent.not_fully_visited-3), prior(prior), move(move), parent(&parent)
            {
                parent.children.push_back(this);
            }

            ~TreeNode() {
                while (!children.empty()) {
                    delete children.back();
                    children.pop_back();
                }
            }

            TreeNode* free(Game::Move m) {
                TreeNode* r = nullptr;

                while (!children.empty()) {
                    auto child = children.back();
                    if (child->move == m) r = child;
                    else delete child; // <----- :)
                    children.pop_back();
                }
                delete this;
                return r;
            }

            nd_c floating_type expectedValue() const {
                if (visits == 0) return 0;
                return (value + total_score_gain*5.0) / floating_type(visits);
            }

            nd_c floating_type score() const {
                return expectedValue() + explorationRate * prior * std::sqrt(parent->visits) / floating_type(visits + 1);
            }

            nd_c TreeNode* selectChild() const {
                TreeNode* best = nullptr;
                floating_type score = 0;

                for (const auto &child : children) {
                    auto s = child->score();
                    if (!best || s > score) {
                        score = s;
                        best = child;
                    }
                }

                return best;
            }

            constexpr void recordVisit(floating_type v, int s) {
                s += score_gain;
                if (parent) {
                    ++parent->visits;
                    if (parent->parent) parent->parent->recordVisit(v, s);
                }
                if (!visits) score_gain = s;
                total_score_gain += s;
                value += v;
                ++visits;
            }

            void addNoise() {
                auto randomness = Utils::Math::dirichlet(dirichletAlpha, children.size(), *Utils::RNG2);
                for (uint i = 0; i < children.size(); ++i) {
                    auto child = children[i];
                    child->prior = (child->prior * weightPrior + randomness[i] * weightDirichlet) / (weightPrior + weightDirichlet);
                }
            }
        };

        Collector<T>* collector;

        Network &network;
        LayerBlock *input, **output;

        uint rollouts = 1900;
        TreeNode* root = nullptr;

        NeuralNetworkTreeSearch(bool side, Network &network, Collector<T>* collector=nullptr):
                Base(side), collector(collector), network(network),
                input(new LayerBlock(*network.input_block, false)), output(new LayerBlock*[network.outputs])
        {
            LayerBlock::getLastOutputBlocks(input, output);
        }

        ~NeuralNetworkTreeSearch() override {
            delete root;
            delete input;
            delete[] output;
        }

        Move suggest() override {
            if (root) root = root->free(Base::board->getLastMove());
            if (!root) {
                root = new TreeNode(Base::board->getLegalMoves().count());
            } else {
                root->parent = nullptr;
            }
            if (!root->visits) {
                root->recordVisit(branchOut(*root), 0);
            }

            root->addNoise();

            int prevDepth = 0;
            std::array<typename Board::BoardChange, Game::AREA> changes{};
            std::array<TreeNode*, Game::AREA> children{};
            for (uint i = 0; i < rollouts; ++i) {
                if (!root->not_fully_visited) break;
                TreeNode* node = root;

                int depth = 0;
                int scoreGain = 0;
                bool same = i, played = false;
                while (node->visits && node->not_fully_visited) {
                    node = node->selectChild();
                    if (same) {
                        if (depth >= prevDepth || node != children[depth]) {
                            same = false;
                            for (int j = prevDepth-1; j >= depth; --j)
                                Base::board->undo(changes[j]);
                        }
                    }
                    if (!same) {
                        played = true;
                        auto pre = Base::board->getScore(Base::board->getTurn());
                        Base::board->play(node->move, &changes[depth]);
                        scoreGain = Base::board->getScore(!Base::board->getTurn()) - pre;
                    }
                    children[depth] = node;
                    depth++;
                }

                if (!node->visits) {
                    auto value = branchOut(*node);
                    if (!node->not_fully_visited) {
                        TreeNode* n = node;
                        while ((n = n->parent)) {
                            if (--n->not_fully_visited) break;
                        }
                    }
                    node->recordVisit(value, scoreGain);
                } node->recordVisit(0, 0);

                if (played) prevDepth = depth;
            }
            for (int j = prevDepth-1; j >= 0; --j) Base::board->undo(changes[j]);

            if ((collector && Base::board->getLegalMoves().count() >= 15)) {
                bool turn = Base::board->getTurn();
                auto e = new Experience<T>{(floating_type)Base::board->getScore(turn)};
                auto in = Base::board->getDecodedState().getPerspective(turn);
                std::copy(in, in + input_size, e->input);
                for (auto child : root->children) {
                    auto move = child->move;
                    auto pos = turn? Game::Position::mirrorHorizontal(move.pos()): move.pos();
                    uint t = turn? (3-move.type()): (move.type()-1);
                    auto v = child->visits/(floating_type)root->visits;
                    e->prior_label[pos*3+t] = v;
                    //if (v) std::cerr << (std::string)move << ": " << v << '\n';
                }
                collector->vec.push_back(e);
            }

            auto move = Move();
            auto visits = 0u;
            floating_type v = -100;
            for (auto child : root->children) {
                if (child->visits > visits || (child->visits == visits && child->expectedValue() >= v)) {
                    move = child->move;
                    visits = child->visits;
                    v = child->expectedValue();
                }
            }
            root = root->free(move);
            return move;
        }

        constexpr floating_type branchOut(TreeNode &node) {
            bool mirror = Base::board->getTurn();
            auto prediction = network.forward(Base::board->getDecodedState().getPerspective(mirror), input, output);
            node.children.reserve(node.not_fully_visited);
            auto legal = Base::board->getLegalMoves();
            for (uint p = 0; p < Game::AREA; ++p) {
                auto pos = mirror? Game::Position::mirrorHorizontal(p).pos: p;
                if (legal.get(pos)) {
                    for (uint t = 1; t < 4; ++t)
                        new TreeNode(node,
                              Game::Move(pos, mirror? (4-t): t),
                              prediction[0][p * 3 + t - 1]
                        );
                }
            }
            return prediction[1][0];
        }
    };
}

template <typename Board>
void benchmark() {
    using namespace Game;
    using std::cout, std::cin, std::cerr;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    uint amount = 50000;
    cerr << "start\n";
    start = std::chrono::system_clock::now();
    Board b;
    typename Board::BoardChange moves[63];
    for (uint i = 0; i < amount; ++i) {
        for (auto &move : moves)
            b.play(b.randomMove(), &move);
        if (b.getPotentialScore(false) || b.getPotentialScore(true)) while (true) cerr << "WAT";
        for (uint j = 62; j < 63; --j)
            b.undo(moves[j]);
    }
    end = std::chrono::system_clock::now();
    cerr << double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / (double)amount << " MILLIs per game" << '\n'; // 4236
    cerr << "done\n";
}

template <typename T>
void benchmarkNetwork(DeepLearning::Network<T> &net, T* in) {
    uint tests = 10000;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::cerr << "starting network benchmark...\n";
    start = std::chrono::system_clock::now();
    for (uint i = 0; i < tests; ++i) {
        net.forward(in);
    }
    end = std::chrono::system_clock::now();
    std::cerr << "finished benchmark:\n";
    std::cerr << double((double)tests / double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) * 1000.0) << " forwards per second" << '\n';
}

template <typename T, typename MF, typename ...Args>
void competition(Args&&... args) {
    using std::cout, std::cin, std::cerr;
    using namespace Game;
    Board<T> board;
    Move m1, m2;

    cin >> m1 >> m2;

    board.play(m1);
    board.play(m2);
    std::string s;
    cin >> s;
    if (s[0] != 'S') {
        board.play(Move(s.c_str()));
    }
    auto bot = new MF(board.getTurn(), std::forward<Args>(args)...);
    bot->board = &board;

    Game::Move move;
    while (!board.isOver()) {
        if (bot->side == board.getTurn()) {
            move = bot->suggest();
            cout << (std::string)move << std::endl;
        } else {
            cin >> move;
        }
        board.play(move);
    }
}

template <typename T>
std::pair<int, int> simulateGame(MoveFinder::MoveController<T>* blue, MoveFinder::MoveController<T>* red) {
    using namespace Game;
    Board<T> board;
    board.play(board.randomMove());
    board.play(board.randomMove());
    blue->board = red->board = &board;

    std::array<MoveFinder::MoveController<T>*, 2> bots{blue, red};
    while (!board.isOver()) {
        board.play(bots[board.getTurn()]->suggest());
    }

    delete blue;
    delete red;

    return {board.getScore(false), board.getScore(true)};
}

template <typename T>
DeepLearning::Network<T>* initRandomNetwork() {
    using namespace DeepLearning;
    using namespace Activation;
    LayerBlock<T>* root = LayerBlock<T>::createLayerBlock({Game::HEIGHT, Game::WIDTH, Game::Board<T>::planes}, {
            {CONV2D,     {3, 3, 8, 1}},
            {ACTIVATION, {}, RELU},
            {CONV2D,     {3, 3, 8, 1}},
            {ACTIVATION, {}, RELU},
            {CONV2D,     {3, 3, 8, 1}},
            {ACTIVATION, {}, RELU},
    }, *Utils::RNG2, 2);
    root->next[0] = LayerBlock<T>::createLayerBlock(root->getOutputSize(), {
            {CONV2D,     {4, 4, 16}},
            {ACTIVATION, {}, RELU},
            {DENSE,      {Game::AREA*3}},
            {ACTIVATION, {}, SOFTMAX},
    }, *Utils::RNG2, 0);
    root->next[1] = LayerBlock<T>::createLayerBlock(root->getOutputSize(), {
            {DENSE,      {64}},
            {ACTIVATION, {}, RELU},
            {DENSE,      {1}},
            {ACTIVATION, {}, TANH},
    }, *Utils::RNG2, 0);
    return new Network<T>(root);
}

template <typename T>
DeepLearning::Network<T>* initNetwork() {
    using namespace DeepLearning;
    std::ifstream f("../networks/network6");
    return Network<T>::load(f);
}

template <typename T>
bool isBetter(DeepLearning::Network<T> &o, DeepLearning::Network<T> &n, uint games=128) {
    using namespace MoveFinder;
    typedef NeuralNetworkTreeSearch<T> Agent;


#ifdef USE_THREADS
    constexpr const uint numThreads = 10;
    std::thread threads[numThreads];
#else
    constexpr const uint numThreads = 1;
#endif
    uint score[2]{}, g = 0;
    auto f = [&games, &g, &o, &n, &score]() {
        while (g < games) {
            if (++g <= games/2) {
                auto s = simulateGame(new Agent(false, o), new Agent(true, n));
                score[0] += s.first;
                score[1] += s.second;
            } else {
                auto s = simulateGame(new Agent(false, n), new Agent(true, o));
                score[1] += s.first;
                score[0] += s.second;
            }
            //std::cerr << s.first << '-' << s.second << '\n';
        }
    };

#ifdef USE_THREADS
    for (auto & thread : threads) thread = std::thread(f);
    for (auto & thread : threads) thread.join();
#else
    f();
#endif

    std::cerr << score[0]/(double)games << '-' << score[1]/(double)games << '\n';
    return score[0]/(double)games-0.4 <= score[1]/(double)games;
}

#ifdef INCLUDE_FILE_SYSTEM
template <typename T>
void train(const std::string &folder) {
    using namespace std::chrono_literals;
    using namespace DeepLearning;
    using namespace MoveFinder;
    typedef NeuralNetworkTreeSearch<T> SmartAgent;
    typedef MinimaxMoveController<BoardEvaluation::PotentialScore, T> MinimaxAgent;

#ifdef USE_THREADS
    constexpr const uint numThreads = 11;
    std::thread threads[numThreads];
#else
    constexpr const uint numThreads = 1;
#endif
    constexpr const uint gamesPerUpdate[2] = {2500, 254};

    const DynamicVector<Loss::Loss<T>> loss = {
            Loss::Loss<T>::template create<Loss::CrossEntropySoftmax>(),
            Loss::Loss<T>::template create<Loss::MSETanh>()
    };
    constexpr const uint miniBatch = 512/numThreads;

    const std::string progressFileName = folder + "/progress.txt";
    const std::string networkFileName = folder + "/network";

    Network<T>* network = nullptr;
    bool running, surpassed;
    uint netID = 0, games = 0, totalGames = 0;
    std::pair<int, int> totalScore = {0, 0};

    std::vector<Experience<T>*> collection[numThreads]{};

    if (!std::filesystem::exists(progressFileName)) {
        std::cerr << "Creating progress file\n";
        std::ofstream f1(progressFileName), f2(networkFileName + "0");
        f1 << "1\n" << netID << "\n0\n" << totalGames;
        f1.close();
        auto n = initRandomNetwork<T>();
        n->save(f2);
        f2.close();
        delete n;
    }
    std::fstream progress(progressFileName, std::ios::in | std::ios::out);
    progress >> running >> netID >> surpassed >> totalGames;
    progress.close();
    progress.clear();

    {
        std::ifstream file(networkFileName + std::to_string(netID));
        network = Network<T>::load(file);
        file.close();

        network->setOptimizer(new Optimizers::SGD<T>(0.00005));
    }

    auto f = [&games, &network, &collection, &gamesPerUpdate, &surpassed, &totalScore](uint thread) {
        while (games < gamesPerUpdate[surpassed]) {
            games += 1 + surpassed;
            Collector<T> col[2]{};
            std::pair<int, int> score;
            if (surpassed) {
                score = simulateGame(new SmartAgent(false, *network, &col[0]), new SmartAgent(true, *network, &col[1]));
            } else {
                score = simulateGame(new SmartAgent(false, *network), new MinimaxAgent(true, &col[0]));
            }
            for (uint i : {0, 1}) {
                col[i].complete((i || !surpassed)? score.second: score.first);
                std::copy(col[i].vec.begin(), col[i].vec.end(), std::back_inserter(collection[thread]));
                if (!surpassed) break;
            }
            //std::cerr << collection[thread].size() << '\n';
            totalScore.first += score.first;
            totalScore.second += score.second;
            std::cerr << games << '/' << gamesPerUpdate[surpassed] << ": " << score.first << '-' << score.second << " (" << thread << ")\n";
        }
    };

    while (running) {
        totalScore = {0, 0};
        games = 0;
        std::cerr << "Games Played: " << totalGames << ", version: " << netID << '\n';

#ifdef USE_THREADS
        for (uint i = 0; i < numThreads; ++i) threads[i] = std::thread(f, i);
        for (auto & thread : threads) thread.join();
#else
        f(0);
#endif
        double avgScore[2] {totalScore.first/(double)gamesPerUpdate[surpassed]*2*surpassed, totalScore.second/(double)gamesPerUpdate[surpassed]*2*surpassed};
        std::cerr << "Avg Score: " << avgScore[0] << '-' << avgScore[1] << "\n";

        if (!surpassed && avgScore[0]-2 >= avgScore[1]) {
            surpassed = true;
            std::cerr << "Surpassed Minimax!\n";
        }

        DataSet<T> set;
        for (auto & i : collection) {
            set.add(i);
            i.clear();
        }
        set.size = set.e.size();
        std::cerr << "Updating network\n";
        network->template train(set, 1, miniBatch, loss, numThreads);

        {
            std::ifstream f1(networkFileName + std::to_string(netID));
            auto old = Network<T>::load(f1);
            f1.close();
            old->setOptimizer(new Optimizers::SGD<T>(0.00005));

            if (isBetter(*old, *network)) {
                std::cerr << "Network did improve! :)\n";
                std::ofstream f2(networkFileName + std::to_string(++netID));
                network->save(f2);
                delete old;

                totalGames += games;
            } else {
                std::cerr << "Network did not improve :(\n";
                delete network;
                network = old;
            }
        }

        std::cerr << "Writing to progress file...\n";

        progress.open(progressFileName);

        progress >> running;
        progress << '\n' << netID << '\n' << surpassed << '\n' << totalGames << '\n';

        progress.close(); progress.clear();
    }

    delete network;
}
#endif

void play(std::initializer_list<Game::Move> moves) {
    using namespace Game;
    Board<double> board;

    for (auto m : moves) {
        board.play(m);
        board.print(std::cerr);
    }

    board.getDecodedState().print(std::cerr);
}

int main() {
    Utils::Random::init();
    using namespace Game;
    using namespace MoveFinder;
    using namespace DeepLearning;

    typedef double T;

#ifdef COMPETITION
    auto network = initNetwork<T>();

    competition<T, NeuralNetworkTreeSearch<T>>(*network);
#else
/*
    std::ifstream f1("networks/network6"), f2("networks/network7");
    auto n1 = Network<T>::load(f1), n2 = Network<T>::load(f2);

    isBetter(*n1, *n2, 250);

    f1.close();
    f2.close();

    return 0;
*/
    /*
    std::ifstream f1("networks/network6");
    auto n1 = Network<T>::load(f1);

    std::ofstream fout("hardcode.txt");

    n1->printHardCodeInitializer(fout);

    f1.close();
    fout.close();
    return 0;
*/

    train<T>("networks");
    return 0;

    std::cerr << "Loading network...\n";
    auto network = initRandomNetwork<T>();
    std::cerr << "Done:\n";
    network->print(std::cerr);

    //simulateGame(new NeuralNetworkTreeSearch<T>(false, *network), new RandomBot<T>(true));
#endif


    std::cerr << "\n\033[m";
    return 0;
}

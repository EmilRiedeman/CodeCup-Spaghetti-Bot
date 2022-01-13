#include <iostream>
#include <chrono>
#include <algorithm>
#include <stack>
#include <vector>
#include <array>

typedef uint_fast8_t uint8;
typedef unsigned int uint;

template <typename T>
using vector_stack = std::stack<T, std::vector<T>>;

#define nd_c [[nodiscard]] constexpr

inline std::ostream &operator<<(std::ostream &out, unsigned char c) {
    return out << (int)c;
}

//#define COLOR_BOARD

namespace Utils {

    template <typename T>
    constexpr T ceilDivide(T a, T b) {
        return 1 + ((a - 1) / b);
    }

    template <typename T>
    constexpr std::pair<T&, T&> minmax(T& x, T& y) {
        return x < y? std::pair<T&, T&>(x, y): std::pair<T&, T&>(y, x);
    }


    struct Random;
    static Random* RNG;

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

        nd_c Position mirrorHorizontal() const {
            return WIDTH-x()-1 + WIDTH*y();
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
            int score[2]{};
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

        constexpr static uint getConnectPlane(uint border, Direction dir) {
            return CONNECTED_GREY1 + (border-1)*4 + dir;
        }

        struct Decoder {
            typedef floating_type* Data;

            constexpr static const uint planes = 16;

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

            constexpr void set(uint plane, uint mirroredPlane, Position pos, floating_type x=1, BoardChange* log=nullptr) const {
                if (log) log->decoded_stack.emplace(plane, mirroredPlane, pos, normal[plane * AREA + pos]);
                normal[plane * AREA + pos] = x;
                mirror[mirroredPlane * AREA + pos.mirrorHorizontal()] = x;
            }

            constexpr void set(uint plane, Position pos, floating_type x=1, BoardChange* log=nullptr) const {
                set(plane, plane, pos, x, log);
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

            void print(std::ostream &out=std::cout) const {
                for (uint plane = 0; plane < planes; ++plane) {
                    if (plane % 4 == 0) out << "\n\n\n";
                    out << plane << ":\n";
                    for (auto arr : {normal, mirror}) {
                        for (uint y = 0; y < HEIGHT; ++y) {
                            for (uint x = 0; x < WIDTH; ++x) {
                                out << arr[plane*AREA + y*WIDTH+x];
                            }
                            out << '\n';
                        }
                        out << '\n';
                    }
                    out << '\n';
                }
            }
        };

        const Decoder _decoded{};

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
                log->score[0] = _score[0];
                log->score[1] = _score[1];
            }
            const uint j0 = pos << 1, j1 = j0 | 1u;
            const uint type = move.type();
            _decoded.set(FREE, pos, 0);
            _decoded.set(type, pos);
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
                print(std::cerr);
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
            _decoded.set(FREE, change.move);
            _decoded.set(_state_grid[change.move], change.move, 0);
            _state_grid[change.move] = 0;
            _legal_moves.orSet(change.move);
            _score[0] = change.score[0];
            _score[1] = change.score[1];
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
    template <typename T>
    class MoveController {
    public:
        const bool side;
    protected:
        Game::Board<T> &board;

        MoveController(Game::Board<T> &board, bool side): board(board), side(side) {}
    public:
        virtual Game::Move suggest() = 0;
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

        MinimaxMoveController(Game::Board<T> &board, bool side): MoveController<T>(board, side) {}

        Game::Move suggest() override {
            uint count = Base::board.getLegalMoves().count();
            if (count <= 25) depth = 4;
            if (count <= 13) depth = 5;
            if (count <= 10) depth = 6;
            if (count <= 9) depth = 7;
            //if (count <= 8) depth = 8;
            Evaluation bestScore(true), alpha(false);
            Game::Move moves[Game::WIDTH * Game::HEIGHT * 3];
            uint size = 0;
            typename Game::Board<T>::BoardChange undo{};
            //std::cerr << "WORD: " << board.getLegalMoves().word << '\n';
            uint i = 0;
            Utils::BitSet64 currentSlice = Base::side? slice2: slice1;
            while (i < Game::WIDTH) {
                //std::cerr << std::bitset<64>(currentSlice.word) << '\n';
                for (auto pos : Base::board.getLegalMoves() & currentSlice) {
                    //std::cerr << (std::string)Game::Position(pos) << ',';
                    for (uint t = 1; t < 4; ++t) {
                        Game::Move m(pos, t);
                        //std::cerr << (std::string)m << '\n';
                        Base::board.play(m, &undo);
                        auto next = minimax<Evaluation>(Base::board, Base::side, depth-1, false, alpha);
                        Base::board.undo(undo);

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
            std::cerr << count << ": board states: " << m_count << " (depth: " << depth << ')' << '\n';
            m_count = 0;
            uint r = Utils::RNG->nextInt(size);
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
        struct TreeSearchNode {
            // todo
        };
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

template <typename Board, typename MF>
void competition(Board &board, MF* bot) {
    using std::cout, std::cin, std::cerr;

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
    board.getDecodedState().print(std::cerr);
}

int main() {
    Utils::Random::init();
    using namespace Game;
    using namespace MoveFinder;
    using std::cout, std::cin, std::cerr;
    typedef float floating_type;
    Board<floating_type> board;

    Move m1, m2;

    cin >> m1 >> m2;

    board.play(m1);
    board.play(m2);
    std::string s;
    cin >> s;
    if (s[0] != 'S') {
        board.play(Move(s.c_str()));
    }
    competition(board, new MinimaxMoveController<BoardEvaluation::PotentialScore, floating_type>(board, board.getTurn()));
    std::cerr << "\n\033[m";
    return 0;
}

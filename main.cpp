#include <iostream>
#include <chrono>
#include <algorithm>
#include <stack>
#include <vector>

typedef uint_fast8_t uint8_t;
typedef uint32_t uint;
template <typename T>
using vector_stack = std::stack<T, std::vector<T>>;

#define ic_func inline constexpr
#define const_ic_func [[nodiscard]] ic_func

inline std::ostream &operator<<(std::ostream &out, unsigned char c) {
    return out << (int)c;
}

//#define COLOR_BOARD

namespace Utils {

    template <typename T>
    ic_func T ceilDivide(T a, T b) {
        return 1 + ((a - 1) / b);
    }

    template <typename T>
    ic_func std::pair<T&, T&> minmax(T& x, T& y) {
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
            RNG = new Random(1200565796);
            std::cerr << *RNG << '\n';
        }
    };

    class BitSet64 {
    public:
        uint64_t word = 0;

        ic_func BitSet64() = default;
        ic_func BitSet64(uint64_t word): word(word) {}

        const_ic_func bool get(uint8_t index) const {
            return (1ull << index) & word;
        }

        ic_func BitSet64& orSet(uint8_t index) {
            word |= (1ull << index);
            return *this;
        }

        ic_func void unset(uint8_t index) {
            word &= ~(1ull << index);
        }

        const_ic_func uint count() const {
            return __builtin_popcountll(word);
        }

        ic_func BitSet64 operator|(const BitSet64 &o) const {
            return word | o.word;
        }

        ic_func explicit operator bool() const {
            return word;
        }

        ic_func BitSet64 &operator|=(const BitSet64 &o) {
            word |= o.word;
            return *this;
        }

        const_ic_func BitSet64 operator^(const BitSet64 &o) const {
            return word ^ o.word;
        }

        template <typename T>
        const_ic_func T sub(uint8_t pos, uint8_t count) const {
            return (T)((word >> pos) & ((1ull << count) - 1ull));
        }

        struct iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type        = uint;

            uint64_t a = 0;
            ic_func void operator++() {
                a ^= -a & a;
            }
            ic_func uint operator*() const {
                return __builtin_ctzll(a);
            }

            ic_func bool operator!=(const iterator &o) const {
                return a != o.a;
            }
        };

        struct reverse_iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type        = uint;

            uint64_t a;
            uint r;
            ic_func reverse_iterator(const BitSet64 &b): a(b.word), r(__builtin_clzll(b.word)) {
            }

            ic_func void operator++() {
                a ^= (1ull << r);
                r = __builtin_clzll(a);
            }

            ic_func uint operator*() const {
                return r;
            }

            ic_func bool operator!=(const iterator &o) const {
                return a != o.a;
            }
        };

        const_ic_func reverse_iterator rbegin() const {
            return {*this};
        }

        const_ic_func reverse_iterator rend() const {
            return {0};
        }

        const_ic_func iterator begin() const {
            return {word};
        }

        const_ic_func iterator end() const {
            return {0};
        }
    };

    template <uint8_t N>
    class TwoBitArray {
    public:
        BitSet64 _words[ceilDivide(N << 1, 64)];
    public:
        ic_func TwoBitArray() = default;

        const_ic_func uint8_t get(uint8_t index) const {
            return _words[index / 32].template sub<uint8_t>(index % 32 * 2, 2);
        }

        ic_func void orSet(uint8_t index, uint8_t v) {
            _words[index / 32].word |= ((uint64_t)v << (index % 32 * 2));
        }

        ic_func void unset(uint8_t index) {
            _words[index / 32].word &= ~(3ull << (index % 32 * 2));
        }

        ic_func void set(uint8_t index, uint8_t v) {
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

    enum Direction: uint8_t {
        UP,
        DOWN,
        LEFT,
        RIGHT
    };

    const_ic_func Direction operator!(Direction dir) {
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
        uint8_t pos = -1;

        ic_func Position(uint x, uint y): pos(y * WIDTH + x) {
        }

        ic_func Position(uint8_t pos): pos(pos) {
        }

        Position() = default;

        ic_func operator uint8_t() const {
            return pos;
        }

        const_ic_func uint8_t x() const {
            return pos % WIDTH;
        }

        const_ic_func uint8_t y() const {
            return pos / WIDTH;
        }

        ic_func Position &translate(Direction d) {
            pos += TRANSLATIONS[d];
            return *this;
        }

        const_ic_func bool isBorder(Direction dir) const {
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
        uint8_t hash = 255;

        ic_func JointPosition(Position p, bool b) {
            hash = (p << 1) | b;
        }

        ic_func JointPosition(uint8_t hash): hash(hash) {
        }

        ic_func JointPosition() = default;

        const_ic_func Position pos() const {
            return hash >> 1;
        }

        const_ic_func bool orientation() const {
            return hash & 1;
        }

        const_ic_func bool null() const {
            return hash == 255;
        }

        ic_func operator uint8_t() const {
            return hash;
        }
    };

    struct Move {

        uint hash = 255;

        ic_func Move() = default;

        ic_func Move(uint a): hash(a) {
        }

        ic_func Move(Position p, uint t) {
            hash = p | (t << 6);
        }

        ic_func Move(const char* str): Move(Position(str[1] - 'a', str[0] - 'a'), str[2] == 'l'? 3: (str[2] == 's')? 2: 1) {
        }

        explicit operator std:: string() const {
            return (std::string)pos() + char(type() == 3? 'l': type() == 2? 's': 'r');
        }

        friend std::istream &operator>>(std::istream &in, Move &move) {
            char c[3];
            in >> c;
            move = std::forward<Move>(Move(c));
            return in;
        }

        const_ic_func Position pos() const {
            return hash & 63u;
        }

        const_ic_func uint8_t type() const {
            return hash >> 6;
        }
    };

    class Board {
    public:
#ifdef COLOR_BOARD
        //                                   RESET  BLUE       RED         BLUE2      RED2  GRAY (5)
        constexpr static const char* COLOR[] {"39", "38;5;17", "38;5;196", "38;5;31", "91", "38;5;240"};
#endif
    private:
        // Union Find:
        constexpr const static uint AIR  = WIDTH * HEIGHT * 2;
        constexpr const static uint WALL[] {AIR+1, AIR+1, AIR+2, AIR+3};
        const_ic_func static Direction getWallDirection(uint wallValue) {
            return static_cast<Direction>(wallValue - WALL[0] + 1);
        }
        struct UnionSet {
            uint8_t borders[2]{};
            JointPosition tips[2]{}, root = -1;
            uint size = 1;
#ifdef COLOR_BOARD
            uint8_t color = 0;
#endif

            UnionSet() = default;
            ic_func UnionSet(const UnionSet&) = default;
            ic_func UnionSet(UnionSet&&) noexcept = default;

            ic_func UnionSet& operator=(UnionSet&&) noexcept = default;

            const_ic_func bool getTip(JointPosition tip) const {
                return tips[0] != tip;
            }

            ic_func void handleBorder(uint8_t border) {
                if (!borders[0]) borders[0] = border;
                else borders[1] = border;
            }

            ic_func void reset() {
                size = 1;
                borders[0] = borders[1] = 0;
                tips[0] = tips[1] = root;
            }

            const_ic_func bool enclosed() const {
                return borders[1];
            }

            ic_func bool operator<(const UnionSet &o) const {
                return size < o.size;
            }
        };

    public:
        struct BoardChange {
            struct UnionParentChange { // todo pair
                JointPosition joint;
                uint8_t parent;

                ic_func UnionParentChange(JointPosition j, uint8_t p): joint(j), parent(p) {
                }
            };

            Position move;
            int score[2]{};
            vector_stack<UnionParentChange> union_parent_stack;
            vector_stack<UnionSet> union_unite_stack;
            vector_stack<std::pair<uint, uint>> p_score_stack;

            BoardChange() = default;
        };
    private:

        const_ic_func uint8_t findUnion(uint8_t pos, BoardChange* change) {
            if (_union_parents[pos] == pos) return pos;
            uint8_t p = findUnion(_union_parents[pos], change);
            if (p != _union_parents[pos]) {
                if (change) change->union_parent_stack.emplace(pos, _union_parents[pos]);
                _union_parents[pos] = p;
            }
            return p;
        }

        const_ic_func uint8_t findUnionConst(uint8_t pos) const {
            if (_union_parents[pos] == pos) return pos;
            return findUnionConst(_union_parents[pos]);
        }

        ic_func UnionSet* unite(JointPosition aTip, uint8_t b, JointPosition bTip, BoardChange* change) {
            uint8_t a = findUnion(aTip, change);
            b = findUnion(b, change);
            auto A = &_union_sets[a], B = &_union_sets[b];
            if (a == b) return A;
            bool bTipI = B->getTip(bTip);
            JointPosition tip = A->tips[!A->getTip(aTip)];

            if (change) {
                if (change->union_parent_stack.empty()
                                || change->union_parent_stack.top().joint != a)
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
        uint8_t _union_parents[WIDTH * HEIGHT * 2ul]{};

        TwoBitArray<WIDTH * HEIGHT> _state_grid;
        int _score[2]{50, 50};
        uint _potential_score_sum[2] = {HEIGHT, HEIGHT}, _potential_score[2][HEIGHT]{};
        BitSet64 _legal_moves = (1ull << (WIDTH * HEIGHT)) - 1ull;
        bool _turn = false, _game_over = false;

        const_ic_func JointPosition neighbour(Position p, Direction d) const {
            if (p.isBorder(d)) return WALL[d];
            return getJoint(p.translate(d), !d);
        }

        ic_func void setPotentialScore(bool side, uint y, uint value, BoardChange* log) {
            if (log) log->p_score_stack.emplace(y + side*HEIGHT, _potential_score[side][y]);
            _potential_score_sum[side] += value - _potential_score[side][y];
            _potential_score[side][y] = value;
        }

        const_ic_func bool getBorderTip(UnionSet &set, Direction border) const {
            auto pos = set.tips[0].pos();
            if (pos.isBorder(border) && ORIENTATIONS[_state_grid.get(pos)][border] == set.tips[0].orientation()) return false;
            return true;
        }
    public:

        static const constexpr Direction SIDES[][4] {
            {},
            {RIGHT, LEFT, UP, DOWN},
            {UP, LEFT, DOWN, RIGHT},
            {RIGHT, LEFT, DOWN, UP},
        };

        static const constexpr uint8_t ORIENTATIONS[][4] {
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

        const_ic_func int getScore(bool s) const {
            return _score[s];
        }

        const_ic_func uint getPotentialScore(bool s) const {
            return _potential_score_sum[s];
        }

        const_ic_func BitSet64 getLegalMoves() const {
            return _legal_moves;
        }

        const_ic_func int getTurn() const {
            return _turn;
        }

        const_ic_func JointPosition getJoint(Position p, Direction dir) const {
            uint8_t t = _state_grid.get(p);
            if (!t) return AIR;
            return {p, static_cast<bool>(ORIENTATIONS[t][dir])};
        }

        Move randomMove(Utils::Random &rand=*Utils::RNG) const {
            uint r = rand.nextInt(_legal_moves.count()), i = 0;
            for (auto pos : _legal_moves) {
                if (i++ == r) return {Position(pos), rand.nextInt(3)+1};
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
            const uint8_t type = move.type();
            _state_grid.orSet(pos, type);
            _legal_moves.unset(pos);

            {
                JointPosition neighbours[4]{};
                UnionSet* sets[] = {&_union_sets[j0], &_union_sets[j1]};
                uint counts[5]{}; // 2-bit index: bit2 = set, bit1 = side; index 4 is extra count
                uint set;
                for (uint i = 0; i < 4u; ++i) {
                    const JointPosition neigh = neighbour(pos, SIDES[type][i]);
                    neighbours[i] = neigh;
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
                        uint count_index = ((i & 1u) << 1) | (border - WALL[2]);
                        if (border > WALL[0]) { // > WALL[0] is blue/red wall
                            if (i & 2u &&
                                    neighbours[(i ^ 1u) & 1u] < AIR &&
                                    neighbours[i] == sets[set ^ 1u]->root) {
                                counts[4] = _union_sets[neighbours[i]].size - counts[count_index ^ 2u] - 1;
                            } else counts[count_index] = _union_sets[neighbours[i]].size;
                        } else if (!set) counts[4] = _union_sets[neighbours[i]].size;

                        sets[set] = unite(j0 | set, neighbours[i], neigh, log);
                    } else sets[set]->handleBorder(neighbours[i]);

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
                        }}
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
                            if (s->borders[0] != WALL[0]) {
                                auto border = static_cast<Direction>(s->borders[0] - WALL[0] + 1);
                                bool i = getBorderTip(*s, border);
                                setPotentialScore(border == RIGHT, s->tips[i].pos().y(), s->size + 1, log);
                            }
                        }
                    }
                }
                if (!log) {
                    std::cerr << _potential_score_sum[0] << " " << _potential_score_sum[1] << "\n";
                    for (uint i = 0; i < HEIGHT; ++i)
                        std::cerr << _potential_score[0][i] << ' ' << _potential_score[1][i] << '\n';
                    std::cerr << std::endl;
                }
            }
            _game_over = !_legal_moves;
            _turn = !_turn;

            if (!log) print(std::cerr);
        }

        void undo(BoardChange &change) {
            _turn = !_turn;
            _game_over = false;
            _state_grid.unset(change.move);
            _legal_moves.orSet(change.move);
            _score[0] = change.score[0];
            _score[1] = change.score[1];
            std::copy(change.score, change.score+2, _score);
            while (!change.p_score_stack.empty()) {
                uint i = change.p_score_stack.top().first;
                setPotentialScore(i >= HEIGHT, i % HEIGHT, change.p_score_stack.top().second, nullptr);
                change.p_score_stack.pop();
            }
            while (!change.union_parent_stack.empty()) {
                _union_parents[change.union_parent_stack.top().joint] = change.union_parent_stack.top().parent;
                change.union_parent_stack.pop();
            }
            while (!change.union_unite_stack.empty()) {
                _union_sets[change.union_unite_stack.top().root] = std::forward<UnionSet>(change.union_unite_stack.top());
                change.union_unite_stack.pop();
            }
            _union_sets[change.move << 1].reset();
            _union_sets[(change.move << 1) | 1u].reset();
        }

        const_ic_func bool isOver() const {
            return _game_over;
        }

        std::ostream &print(std::ostream &out) const {
            constexpr const uint8_t box_width = 5, box_height = 3;
            constexpr const bool grid = false, dots_only = false;
            char r[WIDTH * box_width][HEIGHT * box_height];
            #ifdef COLOR_BOARD
            std::string colors[WIDTH * box_width][HEIGHT * box_height]{};
            constexpr const char* BG = "\033[49m";
            #endif
            for (uint8_t y = 0; y < HEIGHT*box_height; ++y) {
                for (auto & x : r) {
                    x[y] = ' ';
                }
            }
            for (uint8_t y = 0; y < HEIGHT; ++y) {
                for (uint8_t x = 0; x < WIDTH; ++x) {
                    auto type = _state_grid.get(y * WIDTH + x);
                    if (!type) continue;

                    r[x * box_width][y * box_height + 1] = '-';
                    r[x * box_width + 4][y * box_height + 1] = '-';
                    #ifdef COLOR_BOARD
                    colors[x * box_width][y * box_height + 1] = COLOR[_union_sets[findConstUnion(getJoint(Position(x, y), LEFT))].color];
                    colors[x * box_width + 4][y * box_height + 1] = COLOR[_union_sets[findConstUnion(getJoint(Position(x, y), RIGHT))].color];
                    auto c0 = COLOR[_union_sets[findConstUnion(JointPosition(Position(x, y), false))].color];
                    auto c1 = COLOR[_union_sets[findConstUnion(JointPosition(Position(x, y), true))].color];
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
            for (uint8_t x = 0; x < WIDTH; ++x) {
                out << std::string(box_width/2+1, ' ') << char(x+'a') << std::string(box_width/2-!grid, ' ');
            }
            #ifdef COLOR_BOARD
            out << "\033[49m";
            #endif
            out << "\n";
            for (uint8_t y = 0;; ++y) {
                #ifdef COLOR_BOARD
                out << BG;
                #endif
                if (y % box_height == box_height/2) out << ' ' << char((y/box_height) + 'a') << ' ';
                else out << "   ";
                if (!(y % box_height) && (grid || y == HEIGHT * box_height || !y)) {
                    for (uint8_t x = 0; x < WIDTH; ++x) {
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
                for (uint8_t x = 0; x < WIDTH*box_width; ++x) {
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
    class MoveController {
    protected:
        Game::Board &board;
        const bool side;
        MoveController(Game::Board &board, bool side): board(board), side(side) {}
    public:
        virtual Game::Move suggest() = 0;
    };

    namespace BoardEvaluation {
        struct PotentialScore {
            int score;
            uint p_score;

            ic_func explicit PotentialScore(bool maximizing):
                    score(maximizing? INT32_MIN: INT32_MAX),
                    p_score(maximizing? 0: UINT32_MAX) {}

            PotentialScore(const Game::Board &b, bool side):
                    score(b.getScore(side)),
                    p_score(b.getPotentialScore(side)) {}

            PotentialScore() = default;
            PotentialScore(const PotentialScore&) = default;


            const_ic_func bool operator>=(const PotentialScore &o) const {
                return score > o.score || (score == o.score && p_score >= o.p_score);
            }

            const_ic_func bool operator<(const PotentialScore &o) const {
                return !operator>=(o);
            }
        };
    }

    template <typename T>
    T minimax(
            Game::Board &board,
            bool side,
            uint depth,
            bool maximizing=true,
            T alpha=T{false}, T beta=T{true}
            ) {
        typedef T Evaluation;
        if (!depth || board.isOver()) {
            return {board, side};
        }
        Evaluation best(maximizing);
        Game::Board::BoardChange undo;
        for (auto pos : board.getLegalMoves()) {
            for (uint t = 1; t < 4; ++t) {
                board.play({(uint8_t)pos, t}, &undo);
                Evaluation next = minimax(board, side, depth-1, !maximizing, alpha, beta);
                board.undo(undo);

                if (maximizing) {
                    if (best < next) {
                        best = next;
                        if (best >= beta) break;
                        if (best < alpha) alpha = best;
                    }
                } else {
                    if (next < best) {
                        best = next;
                        if (alpha >= best) break;
                        if (beta < best) beta = best;
                    }
                }
            }
        }
        return best;
    }

    template <typename EVALUATOR=BoardEvaluation::PotentialScore>
    class MinimaxMoveController: public MoveController {
    public:
        typedef EVALUATOR Evaluation;

        uint depth = 3;

        MinimaxMoveController(Game::Board &board, bool side): MoveController(board, side) {}

        Game::Move suggest() override {
            uint count = board.getLegalMoves().count();
            if (count == 50) depth = 5;
            if (count == 20) depth = 6;
            if (count == 10) depth = 8;
            if (count == 7) depth = 7;
            Evaluation bestScore(true), alpha(false);
            Game::Move moves[count * 3];
            uint size = 0;
            Game::Board::BoardChange undo;
            //std::cerr << "WORD: " << board.getLegalMoves().word << '\n';
            for (auto pos : board.getLegalMoves()) {
                //std::cerr << (std::string)Game::Position(pos) << ',';
                for (uint t = 1; t < 4; ++t) {
                    Game::Move m(pos, t);
                    board.play(m, &undo);
                    auto next = minimax<Evaluation>(board, side, depth-1, false, alpha);
                    board.undo(undo);

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
            uint i = Utils::RNG->nextInt(size);
            /*
            std::cerr << '\n';
            for (uint j = 0; j < size; ++j) {
                std::cerr << j << ":" << (std::string)moves[j] << ',';
            }
            std::cerr << '\n';
            std::cerr << i << ' ' << size << '\n';
             */
            return moves[i];
        }
    };
}

int main() {
    Utils::Random::init();
    using namespace Game;
    Board board;
/*
    Board::BoardChange changes[WIDTH * HEIGHT];
    uint i = 0;
    while (!board.isOver()) {
        Move m = board.randomMove();
        std::cout << (std::string)m << '\n';
        board.play(m, &changes[i++]);
        board.print(std::cout);
    }

    std::cout << "ROLLING BACK\n\n\n\n\n\n\n\n\n";

    while (i) {
        board.print(std::cout);
        board.undo(changes[--i]);
    }
    board.print(std::cout);

    while (!board.isOver()) {
        Move m = board.randomMove();
        std::cout << (std::string)m << '\n';
        board.play(m);
        board.print(std::cout);
    }

    return 0;*/
    using std::cout, std::cin, std::cerr;

    Move move;
    uint c = 0;

    cin >> move;
    board.play(move);
    cin >> move;
    board.play(move);

    std::string s;
    cin >> s;
    if (s[0] != 'S') {
        move = Move(s.c_str());
        board.play(move);

        c++;
    }
    auto minimax = new MoveFinder::MinimaxMoveController<MoveFinder::BoardEvaluation::PotentialScore>(board, board.getTurn());
    const bool side = board.getTurn();

    while (!board.isOver()) {
        //cerr << ++c << ": " << board.getScore(0) << ' ' << board.getScore(1) << '\n';
        //if (c >= WIDTH * HEIGHT - 63) board.print(cerr) << '\n';
        if (side == board.getTurn()) {
            move = minimax->suggest();
            cout << (std::string)move << std::endl;
        } else {
            cin >> move;
        }
        board.play(move);
    }

    std::cerr << "\n\033[m";
    return 0;
}

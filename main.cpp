#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <stack>
#include <vector>

typedef uint_fast8_t uint8_t;
typedef uint32_t uint;

#define ic_func inline constexpr
#define const_ic_func [[nodiscard]] ic_func

inline std::ostream &operator<<(std::ostream &out, unsigned char c) {
    return out << (int)c;
}

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
            RNG = new Random();
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

    static const constexpr int TRANSFORMATIONS[] {
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

        ic_func Position &transform(Direction d) {
            pos += TRANSFORMATIONS[d];
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

        ic_func Move(Position p, uint8_t t) {
            hash = p | (t << 6);
        }

        ic_func Move(const char* str): Move(Position(str[1] - 'a', str[0] - 'a'), str[2] == 'l'? 3: (str[2] == 's')? 2: 1) {
        }

        explicit operator std:: string() const {
            return std::string{char(pos().y() + 'a'), char(pos().x() + 'a'), char(type() == 3? 'l': type() == 2? 's': 'r')};
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

#define COLOR_BOARD
    class Board {
    private:
        // Union Find:
        constexpr const static uint AIR  = WIDTH * HEIGHT * 2;
        constexpr const static uint WALL[] {AIR+1, AIR+1, AIR+2, AIR+3};
#ifdef COLOR_BOARD
        //                                   RESET  BLUE       RED         BLUE2      RED2  GRAY (5)
        constexpr static const char* COLOR[] {"39", "38;5;17", "38;5;196", "38;5;31", "91", "38;5;240"};
#endif
        struct UnionSet {
            uint8_t borders[2]{};
            uint size = 1;
            uint8_t root;
#ifdef COLOR_BOARD
            uint8_t color = 0;
#endif

            UnionSet() = default;
            constexpr UnionSet(const UnionSet&) = default;
            constexpr UnionSet(UnionSet&&) noexcept = default;

            constexpr UnionSet& operator=(UnionSet&&) noexcept = default;

            void handleBorder(uint8_t border) {
                if (!borders[0]) borders[0] = border;
                else borders[1] = border;
            }

            ic_func void reset() {
                size = 1;
                borders[0] = borders[1] = 0;
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
            struct UnionParentChange {
                JointPosition joint;
                uint8_t parent;

                ic_func UnionParentChange(JointPosition j, uint8_t p): joint(j), parent(p) {
                }
            };

            Position move;
            int score[2]{50, 50};
            std::stack<UnionParentChange, std::vector<UnionParentChange>> union_parent_stack;
            std::stack<UnionSet, std::vector<UnionSet>> union_unite_stack;

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

        const_ic_func uint8_t findConstUnion(uint8_t pos) const {
            if (_union_parents[pos] == pos) return pos;
            return findConstUnion(_union_parents[pos]);
        }

        ic_func UnionSet* unite(uint8_t a, uint8_t b, BoardChange* change) {
            a = findUnion(a, change);
            b = findUnion(b, change);
            auto A = &_union_sets[a], B = &_union_sets[b];
            if (a == b) return A;
            if (*B < *A) { // todo tips
                std::swap(A, B);
                std::swap(a, b);
            }
            if (change) {
                change->union_parent_stack.emplace(a, _union_parents[a]);
                change->union_unite_stack.emplace(*B);
            }
            _union_parents[a] = b;
            B->size += A->size;
            if (A->borders[0]) B->handleBorder(A->borders[0]);
            return B;
        }

        UnionSet _union_sets[WIDTH * HEIGHT * 2ul];
        uint8_t _union_parents[WIDTH * HEIGHT * 2ul]{};

        TwoBitArray<WIDTH * HEIGHT> _state_grid;
        int _score[2]{50, 50};
        BitSet64 _legal_moves = (1ull << (WIDTH * HEIGHT)) - 1ull;
        bool _turn = false, _game_over = false;

        const_ic_func JointPosition neighbour(Position p, Direction d, uint8_t type) const {
            if (p.isBorder(d)) return WALL[d];
            return getJoint(p.transform(d), !d);
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
            {0, 1, 1, 0},
            {0, 0, 1, 1},
            {1, 0, 1, 0}
        };

        Board() {
            for (uint8_t i = 0; i < AIR; ++i) {
                _union_parents[i] = i;
                _union_sets[i].root = i;
            }
        }

        const_ic_func int getScore(bool s) const {
            return _score[s];
        }

        const_ic_func int getTurn() const {
            return _turn;
        }

        const_ic_func JointPosition getJoint(Position p, Direction dir) const {
            uint8_t t = _state_grid.get(p);
            if (!t) return AIR;
            return JointPosition(p, ORIENTATIONS[t][dir]);
        }

        Move randomMove(Utils::Random &rand=*Utils::RNG) const {
            uint r = rand.nextInt(_legal_moves.count()), i = 0;
            for (auto pos : _legal_moves) {
                if (i++ == r) return Move(Position(pos), rand.nextInt(3)+1);
            }
            return {0u};
        }

        void play(Move move, BoardChange* log=nullptr) {
            const Position pos = move.pos();
            if (log) {
                log->move = pos;
                std::copy(_score, _score+2, log->score);
            }
            const uint j0 = pos << 1, j1 = j0 | 1u;
            const uint8_t type = move.type();
            _state_grid.orSet(pos, type);
            _legal_moves.unset(pos);

            {
                JointPosition neighbours[4]{};
                UnionSet* sets[] = {&_union_sets[j0], &_union_sets[j1]};
                uint counts[5]{}; // ab(=11) a = set, b = side
                for (uint i = 0; i < 4u; ++i) {
                    neighbours[i] = neighbour(pos, SIDES[type][i], type);
                    if (neighbours[i] == AIR) continue;
                    if (neighbours[i] < AIR) neighbours[i] = findUnion(neighbours[i], log);
                    uint set = i & 1u; // 0 1 0 1 (boolean)

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
                        uint count_index = ((i & 1u) << 1) + border - WALL[2];
                        if (border > WALL[0]) { // > WALL[0] is colored wall
                            if (i & 2u &&
                                    neighbours[!(i - 2)] < AIR &&
                                    neighbours[i] == sets[!set]->root) {
                                counts[4] = _union_sets[neighbours[i]].size - counts[count_index ^ 2u] - 1;
                            } else counts[count_index] = _union_sets[neighbours[i]].size;
                        } else if (!set) counts[4] = _union_sets[neighbours[i]].size;

                        sets[set] = unite(neighbours[i], j0 | set, log);
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
#endif
                            } else {
#ifdef COLOR_BOARD
                                sets[set]->color = 5;
#endif
                            }
                        } else {
#ifdef COLOR_BOARD
                            sets[set]->color = 5;
#endif
                        }
                    }
#ifdef COLOR_BOARD
                    else if (sets[set]->borders[0] == WALL[0])
                        sets[set]->color = 5;
#endif
                }
            }

            _game_over = !_legal_moves;
            _turn = !_turn;
        }

        void undo(BoardChange &change) {
            // todo test
            _turn = !_turn;
            _game_over = false;
            _state_grid.unset(change.move);
            _legal_moves.orSet(change.move);
            std::copy(change.score, change.score+2, _score);
            while (!change.union_parent_stack.empty()) {
                _union_parents[change.union_parent_stack.top().joint] = change.union_parent_stack.top().parent;
                change.union_parent_stack.pop();
            }
            while (!change.union_unite_stack.empty()) {
                _union_sets[change.union_unite_stack.top().root] = std::move(change.union_unite_stack.top());
                change.union_unite_stack.pop();
            }
            _union_sets[change.move << 1].reset();
            _union_sets[(change.move << 1) | 1].reset();
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

int main() {
    Utils::Random::init();
    using namespace Game;
    Board board;

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

    return 0;
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
    const bool side = board.getTurn();

    while (!board.isOver()) {
        cerr << ++c << ": " << board.getScore(0) << ' ' << board.getScore(1) << '\n';
        if (c >= WIDTH * HEIGHT - 10) board.print(cerr) << '\n';
        if (side == board.getTurn()) {
            move = board.randomMove();
            cout << (std::string)move << std::endl;
        } else {
            cin >> move;
        }
        board.play(move);
    }

    std::cerr << "\n\033[m";
    return 0;
}

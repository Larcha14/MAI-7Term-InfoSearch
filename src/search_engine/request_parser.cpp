#include "request_parser.hpp"


static inline std::string trim_copy(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}


static inline bool is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ;
}

enum class TokType { TERM, AND, OR, NOT, LPAREN, RPAREN };

struct Token {
    TokType type;
    std::string text; 
};


static bool tokenize_query_strict(const std::string& line, std::vector<Token>& out, std::string& err) {
    out.clear();
    err.clear();

    const std::string s = trim_copy(line);
    if (s.empty()) {
        err = "Пустой запрос.";
        return false;
    }

    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];

        if (std::isspace(c)) { ++i; continue; }

        if (c == '(') { out.push_back({TokType::LPAREN, "("}); ++i; continue; }
        if (c == ')') { out.push_back({TokType::RPAREN, ")"}); ++i; continue; }

        if (is_ascii_letter(c)) {
            size_t j = i;
            while (j < s.size() && is_ascii_letter((unsigned char)s[j])) ++j;

            std::string w = s.substr(i, j - i);
            i = j;

            if (w == "AND") out.push_back({TokType::AND, "AND"});
            else if (w == "OR") out.push_back({TokType::OR, "OR"});
            else if (w == "NOT") out.push_back({TokType::NOT, "NOT"});
            else out.push_back({TokType::TERM, std::move(w)});

            continue;
        }


        if (i + 1 < s.size() && is_cyrillic((unsigned char)s[i], (unsigned char)s[i + 1])) {
            size_t j = i;
            while (j + 1 < s.size() && is_cyrillic((unsigned char)s[j], (unsigned char)s[j + 1])) {
                j += 2;
            }
            std::string w = s.substr(i, j - i);
            i = j;

            out.push_back({TokType::TERM, std::move(w)});
            continue;
        }

        err = "Запрещённый символ в запросе на позиции " + std::to_string(i) + ".";
        return false;
    }

    return true;
}

static inline int prec(TokType t) {
    switch (t) {
        case TokType::NOT: return 3;
        case TokType::AND: return 2;
        case TokType::OR:  return 1;
        default: return 0;
    }
}

static inline bool is_op(TokType t) {
    return t == TokType::AND || t == TokType::OR || t == TokType::NOT;
}


static inline bool right_assoc(TokType t) { return t == TokType::NOT; }

static inline void clear_queue(std::queue<std::string>& q) {
    std::queue<std::string> empty;
    q.swap(empty);
}


static bool to_rpn(const std::vector<Token>& toks, std::queue<std::string>& rpn, std::string& err) {
    clear_queue(rpn);
    err.clear();

    std::vector<Token> st;

    enum class Expect { OPERAND, OPERATOR };
    Expect expect = Expect::OPERAND;

    auto pop_ops = [&](TokType incoming) {
        while (!st.empty() && is_op(st.back().type)) {
            TokType top = st.back().type;
            int pt = prec(top), pi = prec(incoming);

            bool pop = (pt > pi) || (pt == pi && !right_assoc(incoming));
            if (!pop) break;

            rpn.push(st.back().text);
            st.pop_back();
        }
    };

    for (const auto& tk : toks) {
        switch (tk.type) {
            case TokType::TERM:
                if (expect != Expect::OPERAND) { err = "Синтаксис: ожидался оператор перед термином."; return false; }
                rpn.push(tk.text);
                expect = Expect::OPERATOR;
                break;

            case TokType::LPAREN:
                if (expect != Expect::OPERAND) { err = "Синтаксис: ожидался оператор перед '('."; return false; }
                st.push_back(tk);
                break;

            case TokType::RPAREN: {
                if (expect != Expect::OPERATOR) { err = "Синтаксис: пустые скобки или оператор перед ')'."; return false; }
                bool ok = false;
                while (!st.empty()) {
                    if (st.back().type == TokType::LPAREN) { ok = true; st.pop_back(); break; }
                    rpn.push(st.back().text);
                    st.pop_back();
                }
                if (!ok) { err = "Синтаксис: лишняя ')'."; return false; }
                expect = Expect::OPERATOR;
                break;
            }

            case TokType::NOT:
                if (expect != Expect::OPERAND) { err = "Синтаксис: NOT должен стоять перед термином или '('."; return false; }
                pop_ops(tk.type);
                st.push_back({TokType::NOT, "NOT"});

                break;

            case TokType::AND:
            case TokType::OR:
                if (expect != Expect::OPERATOR) { err = "Синтаксис: ожидался термин или ')' перед бинарным оператором."; return false; }
                pop_ops(tk.type);
                st.push_back(tk);
                expect = Expect::OPERAND;
                break;

            default:
                err = "Неизвестный токен.";
                return false;
        }
    }

    if (expect != Expect::OPERATOR) { err = "Синтаксис: выражение заканчивается оператором."; return false; }

    while (!st.empty()) {
        if (st.back().type == TokType::LPAREN) { err = "Синтаксис: лишняя '('."; return false; }
        rpn.push(st.back().text);
        st.pop_back();
    }

    return true;
}


static volatile std::sig_atomic_t g_stop = 0;

extern "C" void on_sigint(int) {
    g_stop = 1;
}

bool terminal(std::queue<std::string>& out_rpn) {
    for (;;) {
        if (g_stop) return false;

        std::string line;
        if (!std::getline(std::cin, line)) return false; 


        std::string t = trim_copy(line);
        if (t == "!q") return false;

        if (t.find('!') != std::string::npos) {
            std::cout << "[ERROR] '!' запрещён. Для выхода введи !q.\n";
            continue;
        }

        std::vector<Token> toks;
        std::string err;

        if (!tokenize_query_strict(t, toks, err)) {
            std::cout << "[ERROR] " << err << "\n";
            continue;
        }

        if (!to_rpn(toks, out_rpn, err)) {
            std::cout << "[ERROR] " << err << "\n";
            continue;
        }

        return true; 
    }
}


// int main() {
//     std::vector<std::string> rpn;
//     while (terminal(rpn)) {
//         std::cout << "RPN: ";
//         for (auto& x : rpn) std::cout << x << ' ';
//         std::cout << "\n";
        
//     }
// }

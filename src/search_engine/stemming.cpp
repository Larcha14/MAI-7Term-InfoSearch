#include "stemming.hpp"
#include "utf8_utils.hpp"
#include <string_view>
#include <vector>


bool ends_with(const std::string& w, std::string_view suf) {
    return w.size() >= suf.size() &&
           std::string_view(w).substr(w.size() - suf.size()) == suf;
}

static inline bool is_ru_vowel(std::string_view ch) {
    // гласные: а е и о у ы э ю я:contentReference[oaicite:1]{index=1}
    static constexpr std::string_view V[] = {
        "а","е","и","о","у","ы","э","ю","я"
    };
    for (auto v : V) if (ch == v) return true;
    return false;
}

// RV: после первой гласной. R1/R2: после первого VC, и снова после первого VC в R1 :contentReference[oaicite:2]{index=2}
static void mark_regions(const std::string& w, size_t& rv, size_t& r1, size_t& r2) {
    rv = r1 = r2 = w.size();

    // RV
    {
        size_t p = 0;
        while (p < w.size()) {
            size_t len = utf8_len((unsigned char)w[p]);
            if (p + len > w.size()) break;
            auto ch = std::string_view(w).substr(p, len);
            if (is_ru_vowel(ch)) { rv = p + len; break; }
            p += len;
        }
    }

    // R1
    {
        bool seen_vowel = false;
        size_t p = 0;
        while (p < w.size()) {
            size_t len = utf8_len((unsigned char)w[p]);
            if (p + len > w.size()) break;
            auto ch = std::string_view(w).substr(p, len);
            bool v = is_ru_vowel(ch);
            if (!seen_vowel) {
                if (v) seen_vowel = true;
            } else {
                if (!v) { r1 = p + len; break; }
            }
            p += len;
        }
    }

    // R2 (ищем VC внутри R1)
    {
        bool seen_vowel = false;
        size_t p = r1;
        while (p < w.size()) {
            size_t len = utf8_len((unsigned char)w[p]);
            if (p + len > w.size()) break;
            auto ch = std::string_view(w).substr(p, len);
            bool v = is_ru_vowel(ch);
            if (!seen_vowel) {
                if (v) seen_vowel = true;
            } else {
                if (!v) { r2 = p + len; break; }
            }
            p += len;
        }
    }
}

static inline bool preceded_by_a_or_ya(const std::string& w, size_t suf_start, size_t region_start) {
    // нужно, чтобы буква перед суффиксом была 'а' или 'я' и тоже лежала в RV :contentReference[oaicite:3]{index=3}
    if (suf_start < 2) return false;
    if (suf_start - 2 < region_start) return false;
    std::string_view prev = std::string_view(w).substr(suf_start - 2, 2);
    return (prev == "а" || prev == "я");
}

static bool remove_longest_in_region(std::string& w, size_t region_start,
                                    const std::vector<std::string_view>& suffixes) {
    size_t best = 0;
    for (auto suf : suffixes) {
        if (!ends_with(w, suf)) continue;
        size_t start = w.size() - suf.size();
        if (start >= region_start) {
            if (suf.size() > best) best = suf.size();
        }
    }
    if (best) {
        w.erase(w.size() - best);
        return true;
    }
    return false;
}

static bool remove_longest_in_region_preceded(std::string& w, size_t region_start,
                                             const std::vector<std::string_view>& suffixes) {
    size_t best = 0;
    for (auto suf : suffixes) {
        if (!ends_with(w, suf)) continue;
        size_t start = w.size() - suf.size();
        if (start >= region_start && preceded_by_a_or_ya(w, start, region_start)) {
            if (suf.size() > best) best = suf.size();
        }
    }
    if (best) {
        w.erase(w.size() - best);
        return true;
    }
    return false;
}

// --------- Классы окончаний :contentReference[oaicite:4]{index=4} ----------
static const std::vector<std::string_view> PERFECTIVE_GERUND_1 = {"вшись","вши","в"};
static const std::vector<std::string_view> PERFECTIVE_GERUND_2 = {"ившись","ивши","ив","ывшись","ывши","ыв"};

static const std::vector<std::string_view> REFLEXIVE = {"ся","сь"};

static const std::vector<std::string_view> ADJECTIVE = {
    "ее","ие","ые","ое","ими","ыми","ей","ий","ый","ой","ем","им","ым","ом",
    "его","ого","ему","ому","их","ых","ую","юю","ая","яя","ою","ею"
};

static const std::vector<std::string_view> PARTICIPLE_1 = {"ем","нн","вш","ющ","щ"};
static const std::vector<std::string_view> PARTICIPLE_2 = {"ивш","ывш","ующ"};

static const std::vector<std::string_view> VERB_1 = {
    "ла","на","ете","йте","ли","й","л","ем","н","ло","но","ет","ют","ны","ть","ешь","нно"
};
static const std::vector<std::string_view> VERB_2 = {
    "ила","ыла","ена","ейте","уйте","ите","или","ыли","ей","уй","ил","ыл","им","ым","ен",
    "ило","ыло","ено","ят","ует","уют","ит","ыт","ены","ить","ыть","ишь","ую","ю"
};

static const std::vector<std::string_view> NOUN = {
    "иями","ями","ами","ией","ее", 
    "ев","ов","ие","ье","е","еи","ии","и","ей","ой","ий","й",
    "иям","ям","ием","ем","ам","ом","о","у","ах","иях","ях","ы","ь","ию","ью","ю","ия","ья","я","а"
};

static const std::vector<std::string_view> SUPERLATIVE = {"ейше","ейш"};
static const std::vector<std::string_view> DERIVATIONAL = {"ость","ост"};




std::string stem_word(std::string w) {
    if (w.empty()) return w;

    // ASCII не трогаем 
    if(is_ascii(w[0])) return w;

    size_t rv, r1, r2;
    mark_regions(w, rv, r1, r2);

    // ---- Step 1 ---- :contentReference[oaicite:5]{index=5}
    // perfective gerund
    if (!remove_longest_in_region_preceded(w, rv, PERFECTIVE_GERUND_1)) {
        if (!remove_longest_in_region(w, rv, PERFECTIVE_GERUND_2)) {

            // reflexive
            remove_longest_in_region(w, rv, REFLEXIVE);

            // adjectival: adjective then participle
            if (remove_longest_in_region(w, rv, ADJECTIVE)) {
                // participle after adjective
                // (сначала без условия, потом с условием а/я)
                if (!remove_longest_in_region(w, rv, PARTICIPLE_2)) {
                    remove_longest_in_region_preceded(w, rv, PARTICIPLE_1);
                }
            } else if (!remove_longest_in_region_preceded(w, rv, VERB_1)) {
                if (!remove_longest_in_region(w, rv, VERB_2)) {
                    remove_longest_in_region(w, rv, NOUN);
                }
            }
        }
    }

    // ---- Step 2: удалить конечную "и" ---- :contentReference[oaicite:6]{index=6}
    if (ends_with(w, "и")) {
        size_t start = w.size() - std::string_view("и").size();
        if (start >= rv) w.erase(start);
    }

    // ---- Step 3: DERIVATIONAL в R2 ---- :contentReference[oaicite:7]{index=7}
    remove_longest_in_region(w, r2, DERIVATIONAL);

    // ---- Step 4 ---- :contentReference[oaicite:8]{index=8}
    auto undouble_n = [&]() -> bool {
        if (ends_with(w, "нн")) {
            size_t start = w.size() - std::string_view("нн").size();
            if (start >= rv) {
                // убрать последнюю "н" (2 байта)
                w.erase(w.size() - std::string_view("н").size());
                return true;
            }
        }
        return false;
    };

    if (!undouble_n()) {
        if (remove_longest_in_region(w, rv, SUPERLATIVE)) {
            undouble_n();
        } else if (ends_with(w, "ь")) {
            size_t start = w.size() - std::string_view("ь").size();
            if (start >= rv) w.erase(start);
        }
    }

    return w;
}

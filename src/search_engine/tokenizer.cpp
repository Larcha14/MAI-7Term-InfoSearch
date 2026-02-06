#include "tokenizer.hpp"
#include "utf8_utils.hpp"

int MIN_LETTERS_CNT = 3;
int DECIMAL_PLACES = 2;


bool is_ascii_letter(unsigned char c){
    return (c>= 'a' && c<= 'z') || (c>= 'A' && c<= 'Z');
}

bool is_number(unsigned char c){
    return c>='0' && c<= '9';
}

unsigned char ascii_to_lower(unsigned char c){
    if (c>= 'A' && c<= 'Z'){
        return c - 'A' +'a';
    }
    return c;
}

bool is_valid_utf8(unsigned char c, size_t len, const std::string &s, size_t idx){
    if (idx + len > s.size()) return false;

    for (size_t k = 1; k < len; ++k) {
        unsigned char ck = static_cast<unsigned char>(s[idx + k]);
        if ((ck & 0xC0) != 0x80) return false;
    }


    return true;


}

std::array<unsigned char, 2> utf8_to_lower(unsigned char b1, unsigned char b2){
    // Ё (D0 81) и ё (D1 91) -> е (D0 B5)
    if (b1 == 0xD0 && b2 == 0x81) return {0xD0, 0xB5};
    if (b1 == 0xD1 && b2 == 0x91) return {0xD0, 0xB5};

    if (b1 == 0xD0) {
        // А..П: D0 90..9F -> а..п: D0 B0..BF
        if (b2 >= 0x90 && b2 <= 0x9F) {
            return {0xD0, static_cast<unsigned char>(b2 + 0x20)};
        }
        // Р..Я: D0 A0..AF -> р..я: D1 80..8F
        if (b2 >= 0xA0 && b2 <= 0xAF) {
            return {0xD1, static_cast<unsigned char>(b2 - 0x20)};
        }
        // а..п уже нижний
        return {b1, b2};
    }

    if (b1 == 0xD1) {
        // р..я уже нижний
        return {b1, b2};
    }

    return {b1, b2};
}

std::vector<std::string> tokenize(std::string s){
    std::vector<std::string> out;
    size_t i =0;
    Token token;
    bool add = false;
    // int letter_len = 1;

    while(i<s.size()){
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t letter_len = utf8_len(c);
        if (letter_len == 1){ //if (is_ascii(s[i])){
            
            if (is_ascii_letter(s[i]) or is_number(s[i])){
                token.text.push_back(ascii_to_lower(s[i]));
                token.end++;
                token.letters_cnt++;
            } else{
                add = true;
            }

        } else{
            if (!is_valid_utf8(s[i], letter_len, s, i)){
                add = true;
                letter_len = 1;
                
            } else if (letter_len == 2 && is_cyrillic(c, (unsigned char)s[i+1])){
                auto tmp = utf8_to_lower(c, (unsigned char)s[i+1]);
                token.text.push_back((char)tmp[0]);
                token.text.push_back((char)tmp[1]);
                token.end += letter_len;
                token.letters_cnt++;
            } else{
                add = true;
                letter_len = 1;
            }

        }
        i+=letter_len;
        if (add){
            if (token.size()>= MIN_LETTERS_CNT) out.push_back(token.text);
            add = false;
            token.reset(i);
            
        }
    }

    if (!token.text.empty() && token.size() >= (size_t)MIN_LETTERS_CNT) {
        out.push_back(token.text);
    }

    return out;


}


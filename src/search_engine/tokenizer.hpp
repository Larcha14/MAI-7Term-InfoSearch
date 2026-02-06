#pragma once
#include <string>
#include <vector>
#include <array>

//set:
extern int MIN_LETTERS_CNT;
extern int DECIMAL_PLACES;


struct Token {
    std::string text;   
    size_t start = 0;   
    size_t end   = 0;
    size_t letters_cnt = 0;
    bool is_number(){
        if (size()>0){
            return !text.empty() && text[0] >='0' && text[0] <='9';
        }
        return false;
    }
    void reset(size_t i){
        text.clear();
        start = i;
        end = i;
        letters_cnt = 0;
    };
    size_t size(){
        return letters_cnt;
    }

};

std::vector<std::string> tokenize(std::string s);
bool is_ascii(unsigned char c);
bool is_ascii_letter(unsigned char c);
bool is_number(unsigned char c);
unsigned char ascii_to_lower(unsigned char c);
size_t utf8_len(unsigned char c);
bool is_valid_utf8(unsigned char c, size_t len, const std::string &s, size_t idx);
std::array<unsigned char, 2> utf8_to_lower(unsigned char b1, unsigned char b2);



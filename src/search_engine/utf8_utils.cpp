#include "utf8_utils.hpp"


bool is_ascii(unsigned char c){
    return c <0x80;
}

size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0b110) return 2;
    if ((c >> 4) == 0b1110) return 3;
    if ((c >> 3) == 0b11110) return 4;
    return 1;
}

bool is_cyrillic(unsigned char b1, unsigned char b2) {
    if (b1 == 0xD0) {
        return (b2 == 0x81) || (b2 >= 0x90 && b2 <= 0xBF);
    }
    if (b1 == 0xD1) {
        return (b2 == 0x91) || (b2 >= 0x80 && b2 <= 0x8F);
    }
    return false;
}

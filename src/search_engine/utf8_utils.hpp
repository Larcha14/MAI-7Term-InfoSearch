#pragma once
#include <cstddef>

std::size_t utf8_len(unsigned char c);
bool is_cyrillic(unsigned char b1, unsigned char b2);
bool is_ascii(unsigned char c);
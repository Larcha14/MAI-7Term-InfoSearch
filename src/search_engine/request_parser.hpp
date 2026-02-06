#include "utf8_utils.hpp"
#include <iostream>

#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <queue>
#include <stack>

#include <csignal>
#include <atomic>


bool terminal(std::queue<std::string>& out_rpn);
extern "C" void on_sigint(int);
bool terminal(std::queue<std::string>& out_rpn);
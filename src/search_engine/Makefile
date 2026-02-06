CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic \
  -I/usr/local/include/bsoncxx/v_noabi \
  -I/usr/local/include/mongocxx/v_noabi

LDFLAGS := -L/usr/local/lib64 -L/usr/local/lib
LDLIBS := -lmongocxx -lbsoncxx

BUILDER := builder
SEARCHER := searcher
ALL_TARGETS := $(BUILDER) $(SEARCHER)



COMMON_OBJS := mongoreader.o tokenizer.o stemming.o utf8_utils.o
BUILDER_OBJS := builder.o $(COMMON_OBJS) 
SEARCHER_OBJS := searcher.o $(COMMON_OBJS) request_parser.o


all: $(ALL_TARGETS)

$(BUILDER): $(BUILDER_OBJS)
	$(CXX) $(BUILDER_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(SEARCHER): $(SEARCHER_OBJS)
	$(CXX) $(SEARCHER_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)


searcher.o: searcher.cpp mongoreader.hpp tokenizer.hpp stemming.hpp request_parser.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

builder.o: builder.cpp mongoreader.hpp tokenizer.hpp stemming.hpp
	$(CXX) $(CXXFLAGS) -c builder.cpp -o builder.o

mongoreader.o: mongoreader.cpp mongoreader.hpp utf8_utils.hpp
	$(CXX) $(CXXFLAGS) -c mongoreader.cpp -o mongoreader.o

tokenizer.o: tokenizer.cpp tokenizer.hpp utf8_utils.hpp
	$(CXX) $(CXXFLAGS) -c tokenizer.cpp -o tokenizer.o

stemming.o: stemming.cpp stemming.hpp utf8_utils.hpp
	$(CXX) $(CXXFLAGS) -c stemming.cpp -o stemming.o

request_parser.o: request_parser.cpp request_parser.hpp utf8_utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

utf8_utils.o: utf8_utils.cpp utf8_utils.hpp
	$(CXX) $(CXXFLAGS) -c utf8_utils.cpp -o utf8_utils.o

clean:
	rm -f $(TARGET) $(OBJS)

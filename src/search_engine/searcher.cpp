#include <iostream>
#include <vector>
#include <fstream>
#include <string_view>
#include "request_parser.hpp"
#include "mongoreader.hpp"
#include "tokenizer.hpp"
#include "stemming.hpp"

size_t ID_LEN = 24; //fix!! bsoncxx::oid::to_string()!!

struct elem{
    std::string key;
    std::vector<std::string> values;
};



std::vector<std::string> doc_list_parser(std::string &s){
    std::vector<std::string> v;
    size_t i = 0;

    while (i + ID_LEN<= s.size()){
        
        std::string tmp_str = s.substr(i, ID_LEN);
        i+=ID_LEN+1;
        v.push_back(tmp_str);
    }

    return v;

}

std::vector<std::string> read_postings_csv(const std::string& path, std::string &word){
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    std::string line;

    while (std::getline(f, line)) {
        if (line.empty()) continue;

        if (!line.empty() && line.back() == '\r') line.pop_back();

        // пропускаем заголовок если он есть
        if (line.rfind("term,df,docs", 0) == 0) continue;

        size_t c1 = line.find(',');
        if (c1 == std::string::npos) continue;

        size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) continue;

        std::string term = line.substr(0, c1);
        if (term != word) continue;

        std::string docs = line.substr(c2 + 1);
        return doc_list_parser(docs);
    }

    return {};
}

std::vector<std::string> all_id_list(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    std::string s;
    std::getline(f, s); 

    std::vector<std::string> v = doc_list_parser(s);

    return v;
}


static std::vector<std::string> b_intersect(
    const std::vector<std::string>& a,
    const std::vector<std::string>& b
) {
    std::vector<std::string> out;
    out.reserve(std::min(a.size(), b.size()));
    size_t i=0, j=0;
    while (i<a.size() && j<b.size()) {
        if (a[i]==b[j]) { out.push_back(a[i]); ++i; ++j; }
        else if (a[i]<b[j]) ++i;
        else ++j;
    }
    return out;
}



static std::vector<std::string> b_union(
    const std::vector<std::string>& a,
    const std::vector<std::string>& b
) {
    std::vector<std::string> out;
    out.reserve(a.size() + b.size());
    size_t i=0, j=0;
    while (i<a.size() && j<b.size()) {
        if (a[i]==b[j]) { out.push_back(a[i]); ++i; ++j; }
        else if (a[i]<b[j]) { out.push_back(a[i]); ++i; }
        else { out.push_back(b[j]); ++j; }
    }
    while (i<a.size()) out.push_back(a[i++]);
    while (j<b.size()) out.push_back(b[j++]);
    return out;
}



// A \ B
static std::vector<std::string> b_diff(
    const std::vector<std::string>& a,
    const std::vector<std::string>& b
) {
    std::vector<std::string> out;
    out.reserve(a.size());
    size_t i=0, j=0;
    while (i<a.size() && j<b.size()) {
        if (a[i]==b[j]) { ++i; ++j; }
        else if (a[i]<b[j]) { out.push_back(a[i]); ++i; }
        else { ++j; }
    }
    while (i<a.size()) out.push_back(a[i++]);
    return out;
}


std::vector<std::string> parser(const std::string &path, std::queue<std::string> rpn, std::vector<std::string> &all){
    std::stack<std::vector<std::string>> st;

    if(rpn.empty()) throw std::runtime_error("The request is empty");


    while(!rpn.empty()){
        std::string el = rpn.front();
        rpn.pop();

        if( el == "NOT"){
            auto a = std::move(st.top()); st.pop();

            auto res = b_diff(all, a);
            st.push(std::move(res));


        } else if( el == "AND"){
            auto b = std::move(st.top()); st.pop();
            auto a = std::move(st.top()); st.pop();

            auto res = b_intersect(a, b);
            st.push(std::move(res));

        } else if (el=="OR"){
            auto b = std::move(st.top()); st.pop();
            auto a = std::move(st.top()); st.pop();

            auto res = b_union(a, b);
            st.push(std::move(res));
        } else {
            std::vector<std::string> terms = tokenize(el);
            std::string stem = stem_word(terms[0]);

            st.push(std::move(read_postings_csv(path, stem)));
        }

    }
    return std::move(st.top());
}

static void viewer(std::vector<std::string> answ, MongoDB& db){
    if (answ.empty()) {
        std::cout<<"По вашему запросу ничего не найдено :(\n";
        return;
    } 
    int view_limit = 5;
    int answ_size =answ.size();
    if (answ_size > view_limit) answ_size = view_limit;
    std::cout<<"\n-Результаты поиска-\n"<<"Найдено всего документов: "<<answ.size()<<"\nПервые "<<answ_size<<":\n";
    for(int i =0; i< answ_size;++i){
        try{
            MongoRequest req = db.get_val_by_id(answ[i]);

            std::cout << "\n[" << (i+1) << "] " << req.id << "\n"
                << "title: " << req.title << "\n"
                << "url:   " << req.url << "\n"
                << "url_clean:   " << req.url_clean << "\n"
                << "src:   " << req.source_name << "\n"
                << "snip:  " << req.snippet << "\n";

        } catch (const std::exception& e) {
            std::cout << "\n[" << (i+1) << "] " << answ[i] << "\n"
                      << "ошибка чтения из Mongo: " << e.what() << "\n";
        }


    }



}

int main(){
    std::signal(SIGINT, on_sigint);
    try{
        MongoDB db;
        std::string export_dir   = getenv_valid("EXPORT_DIR");
        std::string doc_fn   = getenv_valid("DOCS_LIST");
        std::string postings_fn = getenv_valid("POSTINGS_FILE");


        std::vector<std::string> all = all_id_list(export_dir + "/" + doc_fn);
        // std::cout<<all[0]<<'\n'<<all[1]<<'\n'<<all[2]<<'\n';

        std::cout<<"Добро пожаловать в поисковик 'как и почему это работает'!\n терминал принимает только слова из кириллицы и латиницы!! Остальные знаки запрещены!!\n Булевые операнды (только в таком регистре): AND OR NOT ) (\n Для Выхода используйте !q или ^C\n Да пребудет с Вами сила!\n";
        std::queue<std::string> rpn;
        while (terminal(rpn)) {
            // std::cout << "RPN: ";
            // for (auto& x : rpn) std::cout << x << ' ';
            // std::cout << "\n";
            viewer(parser(export_dir + "/" +postings_fn,rpn, all), db);

            
        } 



    }
    catch(const std::exception& e){
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    
}
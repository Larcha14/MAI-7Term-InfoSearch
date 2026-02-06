#include <iostream>
#include "mongoreader.hpp"
#include "tokenizer.hpp"
#include "stemming.hpp"
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>


static std::string make_docs_str(const std::vector<std::string>& ids, char delim = '|') {
    std::string out;

    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out.push_back(delim);
        out += ids[i];
    }
    return out;
}

void write_postings_csv(
    const std::string& path,
    std::unordered_map<std::string, std::vector<std::string>>& terms
) {

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    f << "term,df,docs\n";

    for (auto& [term, docs] : terms) {

        std::string docs_joined = make_docs_str(docs, '|');

        f << term << ','
          << docs.size() << ','
          << docs_joined << '\n';
    }
}

void write_zipf_csv(
    const std::string& path,
    const std::unordered_map<std::string, int64_t>& terms_cnt
) {
    std::vector<std::pair<std::string, int64_t>> v;
    v.reserve(terms_cnt.size());
    for (auto& [term, cnt] : terms_cnt) v.emplace_back(term, cnt);

    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) {
                  if (a.second != b.second) return a.second > b.second; // freq desc
                  return a.first < b.first; // term asc for stability
              });

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    f << "rank,term,freq\n";
    for (size_t i = 0; i < v.size(); ++i) {
        f << (i + 1) << ','
          << v[i].first << ','
          << v[i].second << '\n';
    }
}


void write_docs_list(
    const std::string& path,
    const std::vector<std::string>& docs
) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    
    for(size_t i=0; i<docs.size()-1;++i){
        f<<docs[i]<<'|';
    }
    f<<docs[docs.size()-1];

}



int main(){
    std::unordered_map<std::string, std::vector<std::string>> terms_dict;
    std::unordered_map<std::string, int64_t> terms_cnt;

    using clock = std::chrono::steady_clock;

    try{  
        std::string export_dir   = getenv_valid("EXPORT_DIR");
        std::string postings_fn  = getenv_valid("POSTINGS_FILE");
        std::string zipf_fn      = getenv_valid("ZIPF_FILE");
        std::string dock_list_fn = getenv_valid("DOCS_LIST");


        MongoDB db;
        int cnt = db.get_doc_cnt();
        std::cout<<"Cnt: = "<<cnt<<'\n';
        std::vector<std::string> docs(cnt);

        auto t0 = clock::now();
        for(int i=0; i< cnt;++i){
            MongoData val = db.get_val(i);

            docs[i]=(val.id);

            // std::cout<<"i: "<<i<<" | idx = "<<val.id<<"\n"<<val.text<<"\n";
            // std::cout<<"\n\n";
            std::vector<std::string> terms = tokenize(val.text);

            // for(int i=0; i<terms.size(); ++i){
            //     std::cout<< terms[i]<<" ";
            //     if (i %20 == 0){
            //         std::cout<<"\n";
            //     }
            // }
            // std::cout<<"\n\n";
            std::unordered_set<std::string> stems;
            stems.reserve(terms.size());

            for(const auto &t : terms){
                if (t.empty()) continue;
                std::string s = stem_word(t);

                terms_cnt[s]++;

                if (stems.insert(s).second) {
                    terms_dict[s].push_back(val.id);
                }
            }
            if (i!=0 and i%1000==0){
                auto t_tmp = clock::now();
                std::chrono::duration<double> dt_tmp = t_tmp - t0;
                std::cout<< std::fixed << std::setprecision(6)<<"\n[INFO]: Time: "<< dt_tmp.count() << " s | Count of docs: "<<i<<"\n";;
            }

            //УБРАТЬ!!
            // break;
        }
        auto t1 = clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        std::cout << std::fixed << std::setprecision(6)<< "\n[INFO]: Program was finished after: " << dt.count() << " s\n";


        write_postings_csv(export_dir + "/" + postings_fn, terms_dict);
        write_zipf_csv(export_dir + "/" + zipf_fn, terms_cnt);
        write_docs_list(export_dir+"/"+dock_list_fn, docs);



    } catch (const std::exception &e){
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
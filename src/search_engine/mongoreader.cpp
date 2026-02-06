#include "mongoreader.hpp"

std::string getenv_valid(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        throw std::runtime_error(std::string("Missing required env var: ") + name);
    }
    return std::string(v);
}


std::int64_t MongoDB::get_doc_cnt() {
    if (!mongo_client) throw std::runtime_error("Mongo client not initialized");

    auto db  = (*mongo_client)[mongo_db];
    auto col = db[mongo_col];

    bsoncxx::builder::basic::document filter; 
    return col.count_documents(filter.view());
}

mongocxx::instance& mongo_instance() {
    static mongocxx::instance inst{};
    return inst;
}

MongoDB::MongoDB(){
    mongo_instance();

    mongo_uri=getenv_valid("MONGO_URI");
    mongo_db=getenv_valid("MONGO_DB");
    mongo_col=getenv_valid("MONGO_COLLECTION_CLEAN");

    mongo_client.emplace(mongocxx::uri{mongo_uri});

    refresh_index();

}

void MongoDB::refresh_index() {
    ids.clear();
    if (!mongo_client) throw std::runtime_error("Mongo client not initialized");

    try {
        auto db  = (*mongo_client)[mongo_db];
        auto col = db[mongo_col];

        mongocxx::options::find opts;
        opts.sort(make_document(kvp("_id", 1)));
        opts.projection(make_document(kvp("_id", 1)));

        auto cursor = col.find({}, opts);
        for (auto&& doc : cursor) {
            auto e = doc["_id"];
            if (e && e.type() == bsoncxx::type::k_oid) {
                ids.push_back(e.get_oid().value);
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Mongo refresh_index failed: ") + e.what());
    }
}

MongoData MongoDB::get_val(std::size_t i) const {
    if (i >= ids.size()) {
        throw std::out_of_range("MongoDB index out of range");
    }
    return fetch_by_id(ids[i]);
}



MongoData MongoDB::fetch_by_id(const bsoncxx::oid& id) const {
    if (!mongo_client) throw std::runtime_error("Mongo client not initialized");

    auto db  = (*mongo_client)[mongo_db];
    auto col = db[mongo_col];

    auto doc_opt = col.find_one(make_document(kvp("_id", id)));
    if (!doc_opt) {
        throw std::runtime_error("Document not found (collection changed?)");
    }

    auto v = doc_opt->view();

    MongoData out;
    out.id = id.to_string();
    auto html_elem = v["clean_text"];
    out.text = std::string(html_elem.get_string().value);

    return out;
}


static std::string make_snippet_utf8(const std::string& s, size_t max_chars) {
    if (s.empty() || max_chars == 0) return {};

    size_t i = 0;      
    size_t chars = 0;  

    while (i < s.size() && chars < max_chars) {
        size_t len = utf8_len((unsigned char)s[i]);   
        if (i + len > s.size()) break;                
        i += len;
        ++chars;
    }

    std::string out = s.substr(0, i);
    if (i < s.size()) out += "...";
    return out;
}

MongoRequest MongoDB::get_val_by_id(std::string& idx){
    if (!mongo_client) throw std::runtime_error("Mongo client not initialized");

    bsoncxx::oid oid;
    try {
        oid = bsoncxx::oid{idx};
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Bad oid string: '") + idx + "': " + e.what());
    }

    auto db  = (*mongo_client)[mongo_db];
    auto col = db[mongo_col];

    mongocxx::options::find opts;
    opts.projection(make_document(
        kvp("url_clean", 1),
        kvp("url_norm", 1),
        kvp("source_name", 1),
        kvp("title", 1),
        kvp("clean_text", 1) // если хочешь fallback-snippet
    ));

    auto doc_opt = col.find_one(make_document(kvp("_id", oid)), opts);
    if (!doc_opt) {
        throw std::runtime_error("Document not found by _id: " + idx);
    }



    auto v = doc_opt->view();

    MongoRequest out;
    out.id = oid.to_string();
    out.url_clean   = std::string(v["url_clean"].get_utf8().value);
    out.url    = std::string(v["url_norm"].get_utf8().value);
    out.source_name = std::string(v["source_name"].get_utf8().value);
    out.title       = std::string(v["title"].get_utf8().value);

    constexpr size_t SNIP_CHARS = 100;

    out.snippet = make_snippet_utf8(std::string(v["clean_text"].get_utf8().value), SNIP_CHARS);

    return out;





}
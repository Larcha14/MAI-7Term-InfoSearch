#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>

#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>

using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;


#include <cstdlib>   
#include <string>
#include <stdexcept>
#include <vector>
#include <optional>

#include "utf8_utils.hpp"

std::string getenv_valid(const char* name);
mongocxx::instance& mongo_instance();


int get_doc_cnt();

struct MongoData{
    std::string id, text;
};

struct MongoRequest{
    std::string id, url, url_clean, source_name, title, snippet;
};

struct MongoDB{

    explicit MongoDB();
    MongoData get_val(std::size_t i) const;
    std::int64_t get_doc_cnt();

    MongoRequest get_val_by_id(std::string& idx);

    private:
        std::string mongo_uri, mongo_db, mongo_col;
        std::vector<bsoncxx::oid> ids;
        std::optional<mongocxx::client> mongo_client;

    
        MongoData fetch_by_id(const bsoncxx::oid& id) const;
        void refresh_index();

};



from pymongo import MongoClient

client = MongoClient("mongodb://mongo:27017")
db = client["wiki_corpus"]
print(db["documents_clean"].count_documents({}))
print(db["documents_clean"].find_one())
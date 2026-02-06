BOOTSTRAP_SERVICE := bootstrap
MYSQL_SERVICE := mysql
MONGO_SERVICE := mongo
CRAWLER_SERVICE := crawler
SERCHER_SERVICE := searcher
BUILDER_SERVICE := builder

up-mysql:
	docker-compose up -d $(MYSQL_SERVICE)

up-mongo:
	docker-compose up -d $(MONGO_SERVICE)


bootstrap-init: up-mysql
	docker-compose build $(BOOTSTRAP_SERVICE)
	docker-compose run --rm $(BOOTSTRAP_SERVICE)

crawl-init: up-mongo
	docker-compose build $(CRAWLER_SERVICE)
	docker-compose run --rm $(CRAWLER_SERVICE)

verify_mongo: up-mongo
	docker-compose build $(CRAWLER_SERVICE)
	docker-compose run --rm $(CRAWLER_SERVICE) python /app/verify_mongo.py

clean_html: up-mongo
	docker-compose build $(CRAWLER_SERVICE)
	docker-compose run --rm $(CRAWLER_SERVICE) python /app/html_cleaner.py

tmp: up-mongo
	docker-compose build $(CRAWLER_SERVICE)
	docker-compose run --rm $(CRAWLER_SERVICE) python /app/tmp.py

builder: up-mongo
	docker-compose build builder
	docker-compose run --rm  builder

zipf:
	docker-compose build zipf_plot
	docker-compose run --rm zipf_plot

searcher:
	docker-compose build searcher
	docker-compose run --rm searcher

mysql-logs: 
	docker-compose logs -f mysql

crawl-logs:
	docker-compose logs -f crawler

down:
	docker-compose down

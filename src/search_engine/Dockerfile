FROM mongodb/mongo-cxx-driver:latest AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN microdnf upgrade -y \
 && microdnf install -y gcc-c++ make \
 && microdnf clean all

WORKDIR /app

COPY . .

RUN make builder searcher

FROM mongodb/mongo-cxx-driver:latest AS runtime
WORKDIR /app

COPY --from=build /app/builder  /app/builder
COPY --from=build /app/searcher /app/searcher

FROM runtime AS builder
ENTRYPOINT ["./builder"]


FROM runtime AS searcher
ENTRYPOINT ["./searcher"]
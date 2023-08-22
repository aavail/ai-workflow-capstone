FROM ubuntu:latest
LABEL authors="thema"

ENTRYPOINT ["top", "-b"]
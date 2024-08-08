docker build -t binaryalign .
docker tag binaryalign:latest gitlab-ncsa.ubisoft.org:4567/laforge/nlp/binaryalign/binaryalign:latest
docker push gitlab-ncsa.ubisoft.org:4567/laforge/nlp/binaryalign/binaryalign:latest
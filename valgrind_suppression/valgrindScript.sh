echo "remove old log file"
rm minimalraw.log
rm minimal.supp

valgrind --leak-check=full --show-reachable=yes --error-limit=no --suppressions=./suppressionV1.supp --gen-suppressions=all --log-file=minimalraw.log ./oclBuild/tracking-itsu-main ./oneEvent.txt 

cat ./minimalraw.log | ./parse_valgrind_suppressions.sh > minimal.supp


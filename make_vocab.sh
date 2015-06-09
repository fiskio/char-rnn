cat $1 | tr -sc '[:alpha:]' '\n' | sort | uniq -c | sort -rnk 1 > $2 

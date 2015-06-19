LM="th /home/fiskio/advanced-lm/char-rnn/word_pred.lua /home/fiskio/advanced-lm/char-rnn/cv/billion_en/lm_lstm_epoch0.24_1.1731.t7  -pc 0.8 -depth 10 -queue_size 100 -n 20 -min_branch 3 -lmc -vocab /home/fiskio/advanced-lm/char-rnn/data/billion_en/vocab.txt"
OUT="out/big/neural_unigrams/2epoch31"

DATA="gzip -cd /home/fiskio/advanced-lm/char-rnn/data/billion_en/test.txt.gz"
NLINES=1

mkdir -p ${OUT}
time ${DATA} | head -n ${NLINES} | python -m lmchallenge.pred "${LM}" -s --log ${OUT}/pred.log --pretty ${OUT}/pred.txt | tee ${OUT}/pred.out

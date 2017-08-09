# check if both dataset are prepared to use.
if [ ! -d arXiv/abstracts ]; then
	echo "arXiv abstracts does not exist."
	exit 1
fi
if [ ! -d book ]; then
	echo "bookCorpus dataset does not exist."
	exit 1
fi
# let's start!
# cat arXiv/abstracts/* | gshuf | head -525000 > abstracts.txt
# cat book/* | gshuf | head -525000 > book.txt
# cat book.txt abstracts.txt | gshuf > data.txt
# rm book.txt
# rm abstracts.txt

cat arXiv/abstracts/* | gshuf > abstracts.txt
cat book/* | gshuf  > book.txt
head -$1 abstracts.txt > abstracts_pre.txt
head -$1 book.txt > book_pre.txt
cat abstracts_pre.txt book_pre.txt | gshuf > data_pre.txt
rm book_pre.txt
rm abstracts_pre.txt
head -$2 abstracts.txt > abstracts_real.txt
head -$2 book.txt > book_real.txt
cat abstracts_real.txt book_real.txt | gshuf > data.txt
rm book_real.txt
rm abstracts_real.txt
rm book.txt
rm abstracts.txt
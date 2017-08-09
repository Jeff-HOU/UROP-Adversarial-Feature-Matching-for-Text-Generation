# data

I use the same dataset as the author used in his paper, which are:

* [arXiv abstract](https://arxiv.org)
They have a user-unfriendly API with undetailed [docs](https://arxiv.org/help/oa/index)
But you will find a pretty user-friendly tutorial [here](http://betatim.github.io/posts/analysing-the-arxiv/)
My improved version can be found in [here](https://github.com/Jeff-HOU)

BTW, The author claims he got 500M sentences from arXiv abstract, but I only got 100M+.
The earlist abstract that can be accessed is [this article](https://arxiv.org/abs/0704.0001)

* [bookCorpus](http://yknzhu.wixsite.com/mbweb) You need to ask its owner for access right.

To prepare data, run ```bash prepareData.sh $1 $2```
```$1``` is how many sentences would you like to choose from each dataset as debug set.
```$2``` is how many sentences would you like to choose from each dataset as real training set.
# Transformer-pytorch

Pytorch implementation of Google AI's 2017 Transformer model
> 2017 Transformer: Attention Is All You Need
> https://arxiv.org/abs/1706.03762

# Introduction
Google AI in 2017 proposed a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
As you see, this model architecture becomes the base stone of the following state-of-the-art pre-trained models in Natural Language Processing (NLP), such as GPT, BERT, Transformer-XL,XLnet, RoBERTa. 

This repo will walk through the implementation of Transformer. Code is simple, clean and easy to understand. Some of these codes are based on The ![Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

Currently this project is working in process, as always, PRs are welcome :)

# Dependency
* python >= 3.6
* pytorch >=1.0.0
* torchtext >=0.2.3

# Install
~~~
git clone https://github.com/walkacross/transformer-pytorch.git

cd transformer-pytorch

python setup.py develop
~~~

# Quickstart

# Author

Allen Yu (yujiangallen@126.com)

#License
This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Allen Yu, Quantitative Finance Lab, respective Transformer contributors

Copyright (c) 2018 Alexander Rush : The Annotated Trasnformer

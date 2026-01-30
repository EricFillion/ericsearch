<h1 align="center">
  Eric Search
</h1>

<p align="center">
  <img src="images/logo.png" alt="Eric Search logo" width="200">
</p>


A local vector search engine built for speed and scalability. 

- Fast: we use two-level IVF to effectively scale to millions of documents   
- EricRanker(): powered by a cross-encoder model that extracts relevant information from the top documents. 
- Accelerated compute: compatible with both MPS and CUDA. 
- Easy to use: only a few lines of code are needed to train new datasets 
- Lightweight: simple to install and run with a single Python script. 
- Integrated with Hugging Face's Hub.
- Transferable: zip a single folder to move an entire dataset. 


## Install 
```sh
pip install ericsearch
```


## Maintainers
- [Eric Fillion](https://github.com/ericfillion)  Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer

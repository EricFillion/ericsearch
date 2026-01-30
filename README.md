<h1 align="center">
  Eric Search
</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0" height="20">
  </a>

</p>

<p align="center">
  <strong><a href="https://ericsearch.com">ericsearch.com</a></strong>
</p>


A local vector search engine built for speed and scalability. 

- Fast: we use two-level IVF to effectively scale to millions of documents.
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

[Documentation](https://ericsearch.com)

## Maintainers
- [Eric Fillion](https://github.com/ericfillion) Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer 

## Contributing 
We are currently not accepting contributions. 

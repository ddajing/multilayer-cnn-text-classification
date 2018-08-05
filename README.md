# Multilayer CNN Text Classification
This is implementation of Yoon Kim 's https://arxiv.org/abs/1408.5882 paper with Tensorflow.
My code depend on this single layer implementation https://github.com/dennybritz/cnn-text-classification-tf.

## Requirement
- Python 3
- Tensorflow > 0.12
- Using the pre-trained `word2vec` vectors require downloading the binary file from
https://code.google.com/p/word2vec/

### Processing Data
To process the raw data, run

```
python process_data.py path
```

Where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `data.p` in the same folder, which contains the dataset
in the right format.

### Training Model

```
python train.py
```
### References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow)

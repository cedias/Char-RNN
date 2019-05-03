# Just another Pytorch Char-RNN
------

## Usage :

```python
crnn = CharRNN("input.txt") # Create model with input.txt as source (you can load here too)
crnn.load("charnn.chkpt") # while training, the model auto-checkpoints (you can reload)
crnn.train(100) # train for 100 epochs
crnn.generate() # generate a txt
crnn.save("x")  # save for later
```


# Just another Pytorch Char-RNN 
## -> This one is just behind a class for ease of use :)


## Requirements:

- Pytorch
- tqdm

## Usage :


```python
crnn = CharRNN("input.txt",load=None,device='cpu', hidden_size=350, n_layers=1,rnn_cell=nn.LSTM) # Create model with input.txt as source 

crnn.load("charnn.chkpt") # while training, the model auto-checkpoints to charnn.chkpt
crnn.train(iterations=100,chunk_len=110,batch_size=16, print_each=100) # train for 100 batches of size (batch_size,chunk_len) - prints a sample each 100 iterations
txt = crnn.generate(prime_str='A', predict_len=100, temperature=0.8) # generate a txt
crnn.save("char_model")  # save for later
```



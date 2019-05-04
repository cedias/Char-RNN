import unidecode
import string
import random
import re
import torch
import torch.nn as nn

from tqdm import tqdm
import time, math
import torch.nn.functional as f

import unidecode
import string
import random
import re
import torch
import torch.nn as nn

all_characters = string.printable
n_characters = len(all_characters)



class RNN(nn.Module):
    
    def __init__(self,  hidden_size, n_layers=1,rnn_cell=nn.RNN):
        """
        Create the network
        """
        super(RNN, self).__init__()
        all_characters = string.printable
        n_char = len(all_characters)
        self.n_char = n_char
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        #  (batch,chunk_len) -> (batch, chunk_len, hidden_size)  
        self.embed = nn.Embedding(n_char,hidden_size)
        
        # (batch, chunk_len, hidden_size)  -> (batch, chunk_len, hidden_size)  
        self.rnn = rnn_cell(hidden_size,hidden_size,batch_first=True)
        
        #(batch, chunk_len, hidden_size) -> (batch, chunk_len, output_size)  
        self.predict = nn.Linear(hidden_size,n_char)
    
    def forward(self, input):
        """
        batched forward: input is (batch > 1,chunk_len)
        """
        input = self.embed(input)
        output,_  = self.rnn(input)
        output = self.predict(f.tanh(output))
        return output
    
    def forward_seq(self, input,hidden=None):
        """
        not batched forward: input is  (1,chunk_len)
        """
        input = self.embed(input)
        output,hidden  = self.rnn(input.unsqueeze(0),hidden)
        output = self.predict(f.tanh(output))
        return output,hidden
      
    
    

class CharRNN(): 

  
    def __init__(self,filename,load=None,device='cpu', hidden_size=350, n_layers=1,rnn_cell=nn.LSTM):
        self.model = RNN( hidden_size, n_layers=1,rnn_cell=nn.RNN)
        self.device = device
        self.model.to(device)

        self.file = unidecode.unidecode(open(filename).read()) #clean text => only ascii
        self.file_len = len(self.file)
        self.checkpoint_file = "charnn.chkpt"
        print('file_len =', self.file_len)

        if load:
            self.load(load)


    def save(self,path):
        torch.save(self.model.state_dict(), path)

    def load(self,path):
        self.model.load_state_dict(torch.load(path))


    #### GENERATION #####

    def generate(self,prime_str='A', predict_len=100, temperature=0.8):
        prime_input = self.char_tensor(prime_str).squeeze(0)
        hidden = None
        predicted = prime_str+""
        # Use priming string to "build up" hidden state

        for p in range(len(prime_str)-1):
            _,hidden = self.model.forward_seq(prime_input[p].unsqueeze(0),hidden)
                
        #print(hidden.size())
        for p in range(predict_len):
            output, hidden = self.model.forward_seq(prime_input[-1].unsqueeze(0), hidden)
                    # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            #print(output_dist)
            top_i = torch.multinomial(output_dist, 1)[0]
            #print(top_i)
            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            prime_input = torch.cat([prime_input,self.char_tensor(predicted_char).squeeze(0)])

        return predicted

    ########## DATA ##########

    #Get a piece of text
    def random_chunk(self,chunk_len):
        start_index = random.randint(0, self.file_len - chunk_len-1)
        end_index = start_index + chunk_len + 1
        return self.file[start_index:end_index]


    # Turn string into list of longs
    def char_tensor(self,string):
        tensor = torch.zeros(1,len(string),device=self.device).long()
        for c in range(len(string)):
            tensor[0,c] = all_characters.index(string[c])
        return tensor


    #Turn a piece of text in train/test
    def random_training_set(self,chunk_len=200, batch_size=8):
        chunks = [self.random_chunk(chunk_len) for _ in range(batch_size)]
        inp = torch.cat([self.char_tensor(chunk[:-1]) for chunk in chunks],dim=0)
        target = torch.cat([self.char_tensor(chunk[1:]) for chunk in chunks],dim=0)
        
        return inp, target

    #### Training ####
    


    def train_one(self,inp, target):
        """
        Train sequence for one chunk:
        """
        #reset gradients
        self.model_optimizer.zero_grad() 
        
        # predict output
        output = self.model(inp)
        
        #compute loss
        loss =  f.cross_entropy(output.view(output.size(0)*output.size(1),-1), target.view(-1)) 

        #compute gradients and backpropagate
        loss.backward() 
        self.model_optimizer.step() 

        return loss.data.item() 


    def train(self,epochs=1,chunk_len=110,batch_size=16, print_each=100):
        self.model_optimizer= torch.optim.Adam(self.model.parameters())

        with tqdm(total=epochs,desc=f"training - chunks of len {chunk_len}") as pbar:

            for epoch in range(1, epochs + 1):
                loss = self.train_one(*self.random_training_set(chunk_len,batch_size))  #train on one chunk

                if epoch % print_each == 0:
                    self.save(self.checkpoint_file)
                    print("-"*25)
                    print(f"Generated text at epoch {epoch}")
                    print("-"*25)
                    print(self.generate(temperature=0.9))
                    print("-"*25)
                    print(f"model-checkpointed in {self.checkpoint_file}")
                    print("")

                    


                pbar.update(1)
                pbar.set_postfix({"loss":loss})


if __name__ == "__main__":
    crnn = CharRNN("input.txt")
    crnn.load("charnn.chkpt")
    #for chklen in range(500,1500):
    crnn.train(100000) # train for X epochs
    print(crnn.generate())
    crnn.save("model_350")
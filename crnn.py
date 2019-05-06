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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

all_characters = string.printable
n_characters = len(all_characters)


### Copied from AllenNLP library ==> https://allennlp.org/
class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """
    def forward(self, input_tensor):
        # pylint: disable=arguments-differ
        """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


class RNN(nn.Module):
    
    def __init__(self,  hidden_size, n_layers=1,dropout=0.5,rnn_cell=nn.RNN):
        """
        Create the network
        """
        super(RNN, self).__init__()
        all_characters = string.printable
        n_char = len(all_characters)
        self.n_char = n_char
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.dropout = InputVariationalDropout(dropout,inplace=False)

        #  (batch,chunk_len) -> (batch, chunk_len, hidden_size)  
        self.embed = nn.Embedding(n_char,hidden_size)
        
        # (batch, chunk_len, hidden_size)  -> (batch, chunk_len, hidden_size)  
        self.rnns = nn.ModuleList([rnn_cell(hidden_size,hidden_size,batch_first=True) for _ in  range(n_layers)])
        

        #(batch, chunk_len, hidden_size) -> (batch, chunk_len, output_size)  
        self.predict = nn.Linear(hidden_size,n_char)
    
    def forward(self, input):
        """
        batched forward: input is (batch > 1,chunk_len)
        """
        output = self.embed(input)
       
        for rnn in self.rnns:
            do = self.dropout(output)
            output,_ = rnn(do)
        
        output = self.predict(self.dropout(output))
        return output
    
    def forward_seq(self, input,hiddens=None):
        """
        not batched forward: input is  (1,chunk_len)
        """
        output = self.embed(input)
        output = output.unsqueeze(0)
        new_hiddens = []
        
        if hiddens is None:
            hiddens = [None for _ in self.rnns]

        for rnn,hidden in zip(self.rnns,hiddens):
            output,hidden  = rnn(output,hidden)
            new_hiddens.append(hidden)
        output = self.predict(output)
        return output,new_hiddens
      
    
    

class CharRNN(): 

  
    def __init__(self,filename,load=None,device='cpu', hidden_size=512, n_layers=3,rnn_cell=nn.LSTM):
        self.model = RNN( hidden_size, n_layers=n_layers,rnn_cell=rnn_cell)
        self.device = device
        self.model.to(device)

        self.file = unidecode.unidecode(open(filename).read()) #clean text => only ascii
        self.file_len = len(self.file)
        self.tensor_file = self.file2tensor()
        self.checkpoint_file = "charnn.chkpt"
        print('file_len =', self.file_len)

        if load:
            self.load(load)


    def save(self,path):
        torch.save(self.model.state_dict(), path)

    def load(self,path):
        self.model.load_state_dict(torch.load(path,map_location=self.device))


    #### GENERATION #####

    def generate(self,prime_str='.', predict_len=100, temperature=0.8):
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

    #Maps the file to a tensor of longs
    def file2tensor(self):
        all_characters = string.printable
        return torch.LongTensor([all_characters.index(x) for x in self.file]).to(self.device)#,dtype=torch.Long())

    #creates chunks
    def training_set_tensor(self,chunk_len):
        remains = self.tensor_file.size(0)%(chunk_len+1)
        view = self.tensor_file[:-remains].view(-1,chunk_len+1)
        return view
        


    # Turn string into list of longs
    def char_tensor(self,string):
        tensor = torch.zeros(1,len(string),device=self.device).long()
        for c in range(len(string)):
            tensor[0,c] = all_characters.index(string[c])
        return tensor


    #### Training ####
    


    def train_one(self,inp, target):
        """
        Train sequence for one chunk:
        """
        #reset gradients
        self.model.train()
        self.model_optimizer.zero_grad() 
        
        # predict output
        output = self.model(inp)
        
        #compute loss
        loss =  f.cross_entropy(output.view(output.size(0)*output.size(1),-1), target.view(-1)) 

        #compute gradients and backpropagate
        loss.backward() 
        self.model_optimizer.step() 
        self.model.eval()
        return loss.data.item() 


    def train(self,iterations=1,chunk_len=110,batch_size=16, print_each=100):
        
        self.model_optimizer= torch.optim.Adam(self.model.parameters())
        train_file = self.training_set_tensor(chunk_len) 
        data = DataLoader(train_file, batch_size=batch_size,shuffle=True)
        iters = 0

        with tqdm(total=iterations,desc=f"training - chunks of len {chunk_len}") as pbar:
            while (iters < iterations):
                for t in data:
                    tr,te = t[:,:-1].contiguous(),t[:,1:].contiguous()

                    loss = self.train_one(tr,te)  #train on one chunk

                    if iters % print_each == 0:
                        self.save(self.checkpoint_file)
                        print("-"*25)
                        print(f"Generated text at iter {iters}")
                        print("-"*25)
                        print(self.generate(temperature=0.8))
                        print("-"*25)
                        print(f"model-checkpointed in {self.checkpoint_file}")
                        print("")

                        

                    iters += 1
                    pbar.update(1)
                    pbar.set_postfix({"loss":loss})

                    if iters > iterations:
                        break

if __name__ == "__main__":
    crnn = CharRNN("input.txt",device="cpu",rnn_cell=nn.LSTM)
    #print(crnn.training_set_tensor(100))
    crnn.load("charnn.chkpt")
    #for chklen in range(64,100):
    crnn.train(200,batch_size=256,chunk_len=64) # train for X epochs
    #print(crnn.generate())
    crnn.save("ss_512")
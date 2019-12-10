
from cnn import ResNet
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn



class RNN(torch.nn.Module):

    def __init__(self, embed_dim, num_hidden_units, vocab_size, num_layers):
        '''
        Args:
            embed_dim (int) : Embedding dimension between CNN and RNN
            num_hidden_units (int) : Number of hidden units
            vocab_size (int) : Size of the vocabulary
            num_layers (int) : # of layers
        '''

        super(RNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.unit = nn.GRU(embed_dim, num_hidden_units, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden_units, vocab_size)

    def forward(self, cnn_feature, image_caption, caption_size):

        caption_embedding = self.embeddings(image_caption)
        torch_raw_embeddings = torch.cat((cnn_feature.unsqueeze(1), caption_embedding), 1)
        torch_packed_embeddings = nn.utils.rnn.pack_padded_sequence(torch_raw_embeddings, caption_size, batch_first=True)
        torch_packed_embeddings_unit= self.unit(torch_packed_embeddings)[0]
        tokenized_predicted_sentence = self.linear(torch_packed_embeddings_unit[0])

        return tokenized_predicted_sentence

    def sentence_index(self, cnn_feature, beam_size=0):
             
        caption_max_size = 25 
        rnn_hidden_state = None
        rnn_data = cnn_feature.unsqueeze(1)

        # the previous code, which gives the same result as when beam_size=1
        if beam_size == 0: 
            predicted_sentence_idx = []
    
            for idx in range(caption_max_size):
    
                next_state, rnn_hidden_state = self.unit(rnn_data, rnn_hidden_state)
                result_state = self.linear(next_state.squeeze(1))
                predicted_tokenized_word = result_state.max(1)[1]
                predicted_sentence_idx.append(predicted_tokenized_word)
                rnn_data = self.embeddings(predicted_tokenized_word)
                rnn_data = rnn_data.unsqueeze(1)
    
            predicted_sentence_idx = torch.stack(predicted_sentence_idx, 1).squeeze()
    
            return predicted_sentence_idx
        
        # the new code to implement beam search, now only works when batch_size=1
        next_state, rnn_hidden_state = self.unit(rnn_data, rnn_hidden_state)
        result_state = self.linear(next_state.squeeze(1))
        topk_predicted_tokenized_word = result_state.topk(k=beam_size,dim=1)[1]
        
        old_beam_sentence = [] 
        old_beam_word = []
        for k in range(beam_size):
            kth_predicted_tokenized_word = topk_predicted_tokenized_word[:,k]
            old_beam_word.append(kth_predicted_tokenized_word)
            
            
            kth_predicted_sentence_idx = []
            kth_predicted_sentence_idx.append(kth_predicted_tokenized_word)  

            old_beam_sentence.append(kth_predicted_sentence_idx) 

        idx = 1
        while (idx < caption_max_size):
            idx = idx + 1
            new_beam_sentence = []  
            new_beam_word = [] 
            new_beam_prob = []
            for k in range(beam_size):
               
                rnn_data = self.embeddings(old_beam_word[k])
                rnn_data = rnn_data.unsqueeze(1)
                next_state, rnn_hidden_state = self.unit(rnn_data, rnn_hidden_state)
                result_state = self.linear(next_state.squeeze(1))
    
                topk_predicted_tokenized_word = result_state.topk(k=beam_size,dim=1)[1]
                topk_predicted_tokenized_word_prob = result_state.topk(k=beam_size,dim=1)[0] ##
                               
                for j in range(beam_size):
                    previous_predicted_sentence_idx = old_beam_sentence[k].copy()
                    jth_predicted_tokenized_word = topk_predicted_tokenized_word[:,j]
                    jth_sentence = previous_predicted_sentence_idx
                    jth_sentence.append(jth_predicted_tokenized_word) #                                   
                    new_beam_sentence.append(jth_sentence)
                    new_beam_word.append(jth_predicted_tokenized_word)
                    new_beam_prob.append(topk_predicted_tokenized_word_prob[:,j])
                    
            old_beam_sentence = [x for _,x in sorted(zip(new_beam_prob,new_beam_sentence), reverse=True)][0:beam_size]
            old_beam_word = [x for _,x in sorted(zip(new_beam_prob,new_beam_word), reverse=True)][0:beam_size]


        predicted_sentence_idx = old_beam_sentence[0]
        predicted_sentence_idx = torch.stack(predicted_sentence_idx, 1).squeeze()
        return predicted_sentence_idx



   
if __name__ == "__main__":
    
    # Just to show how the rnn works. Never mind what's the input here.
    # Now our beam search only works with batch_size=1.
    
    cnn = ResNet(resnet_version=18)
    rnn = RNN(embed_dim=256, num_hidden_units=512, vocab_size=8000, num_layers=1)
    cnn.eval()
    rnn.eval()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = torchvision.datasets.CIFAR100(root='~/scratch/',
            train=True,download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
    for images, labels in trainloader:
        images = images.to(device)
        cnn_feature = cnn(images)
       
        # previous result
        rnn_tokenized_sentence_prediction = rnn.sentence_index(cnn_feature)
        print(rnn_tokenized_sentence_prediction)
        
        # same as when beam_size=1
        rnn_tokenized_sentence_prediction = rnn.sentence_index(cnn_feature, beam_size=1)
        print(rnn_tokenized_sentence_prediction)
        
        # beam_size=20
        rnn_tokenized_sentence_prediction = rnn.sentence_index(cnn_feature, beam_size=20)
        print(rnn_tokenized_sentence_prediction)


        break
    
    


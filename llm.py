import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.set_float32_matmul_precision('high')
torch.manual_seed(137)
torch.cuda.manual_seed_all(137)


class AttentionModule(nn.Module):

    def __init__(self, num_head, embedding_size):
        super().__init__()
        self.num_head = num_head
        self.embedding_size = embedding_size
        assert self.embedding_size % self.num_head == 0
        self.qkv_matrices = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.qkv_projections = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x):
        batch_size, seq_len, q_dim = x.size()
        qkv = self.qkv_matrices(x)
        q, k, v = qkv.split(self.embedding_size, dim=2)
        k = k.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        q = q.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        v = v.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch_size, seq_len, q_dim) 
        return self.qkv_projections(attn)

class MultiLayerPerceptron(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.activation = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, num_head, embedding_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = AttentionModule( 
                                        num_head=num_head, 
                                        embedding_size=embedding_size
                                        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.multi_layer_perceptron = MultiLayerPerceptron(
                                        embedding_size=embedding_size
                                        )

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.multi_layer_perceptron(self.layer_norm2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, seq_len, vocab_size, block_cnt, num_head, embedding_size):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.block_cnt = block_cnt
        self.num_head = num_head
        self.embedding_size = embedding_size

        self.positional_embeddings = nn.Embedding(self.seq_len, self.embedding_size)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(num_head=num_head, embedding_size=embedding_size) for _ in range(self.block_cnt)]) 
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.unembed = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        self.word_embeddings.weight = self.unembed.weight

        self.apply(self.initialize_weights) # initialize weights

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.0144 / self.block_cnt ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, labels=None):
        batch_size, seq_len = token_ids.size()
        position_idxs = torch.arange(0, seq_len, dtype=torch.long, device=token_ids.device) 
        x = self.word_embeddings(token_ids) + self.positional_embeddings(position_idxs)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.unembed(x) 
        loss = None
        if labels is not None:
            input = logits.view(-1, logits.size(-1))
            target = labels.view(-1)
            loss = F.cross_entropy(input, target)
        return logits, loss

class DataLoader:
    def __init__(self, batch_size, seq_len, num_processes, split):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_processes = num_processes
        self.split = split
        assert split in ['train', 'validation', 'test'], f"invalid split: {split}"
        files = os.listdir(data_dir)
        files = [filename for filename in files if ((split in filename) and filename.endswith(".npy"))]
        files = [os.path.join(data_dir, s) for s in files]
        self.shards = files
        self.shard_cnt = len(self.shards)
        assert len(files) >= 1, f"found {len(files)} files for split {split}"
        self.current_shard = 0
        self.choose_shard_and_position()

    def load_tokens(self, filename):
        idxs_array = np.load(filename)
        idxs_array = idxs_array.astype(np.int32) # Long array of integers
        idxs_tensor = torch.tensor(idxs_array, dtype=torch.long) # Long tensor of integers
        return idxs_tensor

    def next_batch(self):
        batch_size, seq_len = self.batch_size, self.seq_len
        buf = self.tokens[self.current_position : self.current_position+batch_size*seq_len+1]
        x = (buf[:-1]).view(batch_size, seq_len)
        y = (buf[1:]).view(batch_size, seq_len)
        self.current_position += batch_size * seq_len * self.num_processes
        if self.current_position + (batch_size * seq_len * self.num_processes + 1) > len(self.tokens):
            # The current shard is exhausted, so we need to load the next one
            self.choose_shard_and_position()
        return x, y

    def choose_shard_and_position(self):
        if self.split == "train":
            # For training, we want to move from one shard to the other, and 
            # randomize the position
            self.current_shard = (self.current_shard + 1) % self.shard_cnt
            self.current_position = np.random.randint(0, self.batch_size - 1)
        else:
            # For validation and testing, we always want to start from the 
            # beginning of the first shard
            self.current_shard = 0
            self.current_position = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])

vocab_size = 50304 
block_cnt = 12 
seq_len = 4 
batch_size = 2
total_batch_size = 16
num_head = 12 
embedding_size = 768 
max_learning_rate = 1e-4
min_learning_rate = max_learning_rate * 0.1 # TODO make lr vary
warmup_steps = 715
max_steps = 20

data_dir = "FNSPID_transformed"

device = "cpu"
evaluate_on_hellaswag = False
printing_freq = 1 # Print results every X steps
saving_model_freq = 2 # Saves torch model every X steps 

tokenizer = tiktoken.get_encoding("gpt2")
test_gen_tokens = tokenizer.encode("The NASDAQ 100 Pre-Market Indicator is up")

assert total_batch_size % (batch_size * seq_len) == 0
grad_cumul_steps = total_batch_size // (batch_size * seq_len)

# Prefix for saving artifacts
import datetime
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

train_loader = DataLoader(batch_size=batch_size, seq_len=seq_len, num_processes=1, split="train")
validation_loader = DataLoader(batch_size=batch_size, seq_len=seq_len, num_processes=1, split="validation")


model = Transformer(seq_len=seq_len, vocab_size=vocab_size, block_cnt=block_cnt, num_head=num_head, embedding_size=embedding_size)
model.to(device)

# TODO experiment with this
def get_learning_rate(step, max_learning_rate, min_learning_rate):
    if step < warmup_steps:
        return max_learning_rate * (step+1) / warmup_steps
    elif step > max_steps:
        return min_learning_rate
    else:
        decay = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

# TODO experiment with regularization
def create_optimizer(model, weight_decay, learning_rate):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if True:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer

params = {param for _, param in model.named_parameters() if param.requires_grad}
optimizer = torch.optim.Adam(params, lr=max_learning_rate, betas=(0.9, 0.99))

# Create a log
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: 
    pass # clears contents

training_loss = [] # (step, train_loss)
validation_loss = [] # (step, valid_loss)
for step in range(max_steps):
    last_step = (step == max_steps - 1)

    # Evaluate in validation dataset every printing_freq steps
    if step % printing_freq == 0 or last_step:
        model.eval()
        validation_loader.choose_shard_and_position()
        with torch.no_grad():
            validation_loss_cumul = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = validation_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                validation_loss_cumul += loss.detach()
        validation_update_msg = f"Validation loss before step {step} is {validation_loss_cumul.item():.6f}"
        print(validation_update_msg)
        validation_loss.append((step, validation_loss_cumul.item()))
        with open(log_file, "a") as f:
            f.write(validation_update_msg)

    # Save the model every saving_model_freq steps
    if (step % saving_model_freq == 0 or last_step) and step > 0:
        model_path = os.path.join(log_dir, f"{now_str}_model_{step:05d}.pth")
        torch.save(model, model_path)

    # TODO generate from the model! 

    # Forward pass
    model.train()

    # Backward pass
    optimizer.zero_grad()
    train_loss_cumul = 0.0
    for _ in range(grad_cumul_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_cumul_steps
        train_loss_cumul += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = max_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    tokens_processed = train_loader.batch_size * train_loader.seq_len * grad_cumul_steps * 1
    train_update_msg = f"Training loss before step {step+1} is {train_loss_cumul.item():.6f}"
    training_loss.append((step+1, train_loss_cumul.item()))
    with open(log_file, "a") as f:
        f.write(train_update_msg)

# Save the lists of training loss and valid
np.save(os.path.join(log_dir, f'{now_str}_training_loss.npy'), np.array(training_loss))
np.save(os.path.join(log_dir, f'{now_str}_validation_loss.npy'), np.array(validation_loss))

# TODO explain why accuracy doesn't really matter 

# Plot train and validation performance
train_steps, train_losses = zip(*training_loss)
val_steps, val_losses = zip(*validation_loss)

plt.figure(figsize=(8, 6))
plt.plot(train_steps, train_losses, label='Train Loss', color='blue')
plt.plot(val_steps, val_losses, label='Validation Loss', color='orange')
plt.xticks(train_steps)  # or set custom ticks if desired
plt.xlabel('Steps', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss by Step', fontsize=16)
plt.legend(fontsize=16)
plt.show()

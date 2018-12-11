
# coding: utf-8

# In[12]:


from utils import *
from network import *


# In[13]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[14]:


weights = load_pretrained_weights()


# In[15]:


model = VggVox(weights=weights)
model = model.to(device)


# In[16]:


criterion = ContrastiveLoss()
criterion = criterion.to(device)


# In[17]:


loss_list = []
best_loss = torch.autograd.Variable(torch.tensor(np.inf)).float()


# In[18]:


LEARNING_RATE = 1e-3
N_EPOCHS = 10
BATCH_SIZE = 32


# In[19]:


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[9]:


# model, _, optimizer = load_saved_model("checkpoint_20181211-030043_0.014894404448568821.pth.tar", test=False)


# In[20]:


voxceleb_dataset = VoxCelebDataset(PAIRS_FILE)
train_dataloader = DataLoader(voxceleb_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
n_batches = int(len(voxceleb_dataset) / BATCH_SIZE)

print("training unique users", len(voxceleb_dataset.training_users))
print("training samples", len(voxceleb_dataset))
print("batches", int(len(voxceleb_dataset) / BATCH_SIZE))


# In[11]:


for epoch in range(1, N_EPOCHS+1):
    running_loss = torch.zeros(1)

    for i_batch, data in enumerate(train_dataloader, 1):
        mfcc1, mfcc2, label = data['spec1'], data['spec2'], data['label']
        mfcc1 = Variable(mfcc1.float(), requires_grad=True).to(device)
        mfcc2 = Variable(mfcc2.float(), requires_grad=True).to(device)
        label = Variable(label.float(), requires_grad=True).to(device)

        output1, output2 = model(mfcc1.float(), mfcc2.float())

        optimizer.zero_grad()

        loss = criterion(output1, output2, label.float())

#         assert mfcc1.dim() == mfcc2.dim() == 4
#         assert output1.dim() == output2.dim() == 2
#         assert loss.requires_grad and output1.requires_grad and output2.requires_grad
#         assert loss.grad_fn is not None and output1.grad_fn is not None and output2.grad_fn is not None

#         print("loss", loss, loss.requires_grad, loss.grad_fn)
#         print("output1", output1.shape, output1.requires_grad, output1.grad_fn, output1.device)
#         print("output2", output2.shape, output2.requires_grad, output2.grad_fn, output2.device)

        loss.backward()

#         assert mfcc1.requires_grad and mfcc2.requires_grad
#         for name, param in model.named_parameters():
#             assert param.requires_grad and param.grad is not None, (name, param.requires_grad, param.grad)

        optimizer.step()

        loss_list.append(loss.item())
        running_loss += loss.item()
        if i_batch % int(n_batches/20) == 0:
            print("Epoch {}/{}  Batch {}/{} \nCurrent Batch Loss {}\n".format(epoch, N_EPOCHS, i_batch, n_batches, loss.item()))

    epoch_loss = running_loss / len(voxceleb_dataset)
    print("==> Epoch {}/{} Epoch Loss {}".format(epoch, N_EPOCHS, epoch_loss.item()))

    is_best = epoch_loss < best_loss
    if is_best:
        best_loss = epoch_loss

        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        loss=epoch_loss)
    else:
        print("### Epoch Loss did not improve\n")

#     plt.plot(loss_list[50:])
#     plt.show()


# In[ ]:


# plt.plot(loss_list[5000:])


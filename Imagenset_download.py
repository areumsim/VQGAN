### https://www.gpters.org/c/ai-developers/using-huggingface-dataset

#%%
from datasets import load_dataset
img_data = load_dataset("evanarlian/imagenet_1k_resized_256", 
                        cache_dir='./ImageNet_1k_256')


# %%
print(img_data)

# DatasetDict({
#     train: Dataset({
#         features: ['image', 'label'],
#         num_rows: 1281167
#     })
#     val: Dataset({
#         features: ['image', 'label'],
#         num_rows: 50000
#     })
#     test: Dataset({
#         features: ['image', 'label'],
#         num_rows: 100000
#     })
# })

#%%
print(dataset['train'])

# >> Dataset({
#     features: ['image', 'label'],
#     num_rows: 5328
# })

train_img_data = img_data['train']['image']
print(type(train_img_data)) # <class 'list'>
print(type(train_img_data[0])) # PIL.JpegImagePlugin.JpegImageFile
train_img_data[100].show()

print(img_data['train']['label'][:5]) # [6, 3, 11, 16, 1]
print(img_data['train']['label'][100]) # 1 
print(img_data['train'].features)
# {'image': Image(decode=True, id=None), 
#  'label': ClassLabel(names=['burger', 'butter_naan', 'chai', 
# 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice',
#  'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi',
# 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'], id=None)}


#%%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("ImageNet_1k_256", split='train')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
dataset = dataset.map(lambda e: tokenizer(e['sentence1']), batched=True)

dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# %%
next(iter(dataloader))

img = dataloader[0] # torch.Size([3, 320, 320]) / (19, 5)
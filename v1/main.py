
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
# from lion_pytorch import Lion
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

import wandb
from ConvAutoencoder import ConvAutoencoder
from dataloader_v1 import COCODataset, show_originalimage

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    # CUDA Version: 12.2
    # print(torch.__version__)  # 2.0.1+cpu -> 2.1.0+cu121

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")


    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    batch_size = cfg['train_params']["batch_size"]
    num_epoch = cfg['train_params']["num_epoch"]


    # # ###### start a new wandb run to track this script ######
    # wandb.init(
    #     project="autoencoder-pytorch",
    #     name = "areumsim",
    #     # # track hyperparameters and run metadata
    #     # config={
    #     # "learning_rate": cfg['train_params']["learning_rate"],
    #     # "architecture": "ResNet50",
    #     # "dataset": "coco",
    #     # "epochs": num_epoch,
    #     # }
    #     # mode="disabled"
    # )
    # # 실행 이름 설정
    # wandb.run.name = 'First wandb _ train with tr '
    # wandb.run.log_code(".")
    # wandb.run.save()



    # ###### dataload & backbond - coco ######
    coco_train = COCODataset(
        cfg['data']
    )
    loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=True)
    # image = coco_train[0]

    ###### model ######
    convAE_model = ConvAutoencoder(cfg).cuda()

    ###### optimizer and loss ######
    # Get all parameters from the main model
    all_parameters = set(convAE_model.parameters())

    optimizer = torch.optim.AdamW(all_parameters, lr=cfg['train_params']["learning_rate"], weight_decay=cfg['train_params']['weight_decay'])
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    epoch_losses = [] 
    for epoch in range(num_epoch):
        bar = tqdm(loader_train)
        batch_losses = []
        for i_batch, (image) in enumerate(bar):
            images = image.to(device).float()  # [b, 3, 320, 320] : b c H W
            # labels = label.to(device).float()

            optimizer.zero_grad()
            outputs = convAE_model(images)
            loss = criterion(outputs, images)
            
            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())
            # print(f"loss: {loss.item()}")
            
            batch_losses.append(loss.item())
                
            # log metrics to wandb
            # n_iter = epoch * len(loader_train) + i_batch + 1
            # wandb.log({"loss": loss.item(),
            #            'iter':n_iter}, step=n_iter) 

            # for evaluation, draw a predicted bbox and class label
            if i_batch % 100 == 0:
                img_true = show_originalimage(images[0])
                img_predict = show_originalimage(outputs[0].detach())
                
                # log images to wandb
                # wandb.log({
                #     'images': wandb.Image(images[0]),
                #     'prediction result': wandb.Image(outputs[0])
                # })

                # save images and outputs as one image file on the local folder
                concatenated_image = np.concatenate((img_true, img_predict), axis=1)
                plt.imshow(concatenated_image)
                plt.savefig(f"./result_image/combined_image_e{epoch}_b{i_batch}.png")
                plt.clf()            

            # save model's checkpoint every 5000 batch
            if i_batch % 5000 == 0:
                torch.save(convAE_model.state_dict(), f"./result_model/convAE_model_e{epoch}_b{i_batch}.pth")

        # save and show loss
        lss = np.mean(batch_losses)
        epoch_losses.append(lss)
        # plt.plot(np.array(loss), 'r')

        plt.plot(np.arange(len(epoch_losses)), epoch_losses, marker='.', c='red', label='Trainset_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.show()
        plt.savefig(f"./result_image/loss_{epoch}.png")
        plt.close()
        
        with open(f"./result_model/loss.txt", 'w+') as f:
            f.write('\n'.join(map(str, str(lss))))
    
        

    ## save model and loss
    torch.save(convAE_model.state_dict(), f"./result_model/convAE_model_final({epoch})_0219_1.pth")
    torch.save(loss, f"./result_model/loss_final({epoch}).txt")

    wandb.finish()


## loss 를 따로 저장. (classification / box l1 loss / giou loss )
## class가 80인 애들은 box를 안그리기

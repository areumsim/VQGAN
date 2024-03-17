
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
    wandb.init(
        project="autoencoder-pytorch",
        name = "areumsim",
        # # track hyperparameters and run metadata
        # config={
        # "learning_rate": cfg['train_params']["learning_rate"],
        # "architecture": "ResNet50",
        # "dataset": "coco",
        # "epochs": num_epoch,
        # }
        # mode="disabled"
    )
    # 실행 이름 설정
    wandb.run.name = 'update recon., loss, pooling'
    wandb.run.log_code(".")
    wandb.run.save()



    # ###### dataload & backbond - coco ######
    # coco_train = COCODataset(
    #     cfg['data']
    # )
    # loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=True)
    # # image = coco_train[0]

    ###### dataload & backbond - coco ######
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -1과 1 사이의 값으로 정규화
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet의 mean/std로 정규화
        ]
    )
    # img_dir = 'C:/Users/wolve/arsim/autoencoder/CIFAR10/'
    # train = datasets.CIFAR10(root=img_dir, train=True, download=True, transform=transform)
    # test = datasets.CIFAR10(root=img_dir, train=False, download=True, transform=transform)

    img_dir = 'C:/Users/wolve/arsim/autoencoder/STL10/' 
    train = datasets.STL10(root=img_dir, split='train', download=True, transform=transform) #5000
    test = datasets.STL10(root=img_dir, split='test', download=True, transform=transform)   #8000

    data_loader_tr = torch.utils.data.DataLoader(train,
                                            batch_size=cfg['train_params']['batch_size'],
                                            shuffle=True)
    data_loader_ts = torch.utils.data.DataLoader(train,
                                            batch_size=cfg['train_params']['batch_size'],
                                            shuffle=True)

    ###### model ######
    convAE_model = ConvAutoencoder(cfg).cuda()

    ###### optimizer and loss ######
    # Get all parameters from the main model
    all_parameters = set(convAE_model.parameters())

    optimizer = torch.optim.AdamW(all_parameters, lr=cfg['train_params']["learning_rate"], weight_decay=cfg['train_params']['weight_decay'])
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #TODO: L1 ? L2 ? / L1 + perceptual loss

    epoch_losses = [] 
    epoch_test_losses = [] 
    for epoch in range(num_epoch):
        bar = tqdm(data_loader_tr)
        batch_losses = []
        for i_batch, (image, _) in enumerate(bar):
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
                
            ## log metrics to wandb
            # n_iter = epoch * len(data_loader_tr) + i_batch + 1
            # wandb.log({"loss": loss.item(),
            #            'iter':n_iter}, step=n_iter) 

            # for evaluation, draw a predicted bbox and class label
            if i_batch % 5000 == 0:
                img_true = show_originalimage(images[0])
                img_predict = show_originalimage(outputs[0].detach())
                
                ## log images to wandb
                wandb.log({
                    'train images': wandb.Image(img_true),
                    'train predict': wandb.Image(img_predict)
                })

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

        # plt.plot(np.arange(len(epoch_losses)), epoch_losses, marker='.', c='red', label='Trainset_loss')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # # plt.show()
        # plt.savefig(f"./result_image/loss.png")
        # plt.close()
        
        # with open(f"./result_model/loss.txt", 'w+') as f:
        #     f.write('\n'.join(map(str, str(lss))))

        ### evaluate(model, test_loader):
        convAE_model.eval()
        test_loss = []
        correct = 0
        with torch.no_grad():
            for image, label in data_loader_ts:
                image = image.to(device)
                output = convAE_model(image)
                test_loss.append(criterion(output, image).item())
   
        img_true = show_originalimage(image[0])
        img_predict = show_originalimage(output[0].detach())
        concatenated_image = np.concatenate((img_true, img_predict), axis=1)
        
        ## log images to wandb
        wandb.log({
            'test images': wandb.Image(img_true),
            'test predict': wandb.Image(img_predict)
        })

        plt.imshow(concatenated_image)
        plt.savefig(f"./result_image/combined_image_test_e{epoch}.png")
        plt.clf()   

               
        test_loss = np.mean(test_loss)
        epoch_test_losses.append(test_loss)
             
        ## log metrics to wandb
        wandb.log({"tr loss": epoch_losses[-1],
                   "ts loss": epoch_test_losses[-1],
                    'iter':epoch}) 
        
        plt.plot(np.arange(len(epoch_losses)), epoch_losses, marker='.', c='red', label='Train_loss')
        plt.plot(np.arange(len(epoch_losses)), epoch_test_losses, marker='.', c='blue', label='Test_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        # plt.show()
        plt.savefig(f"./result_image/loss.png")
        plt.close()
        
        with open(f"./result_model/loss.txt", 'w+') as f:
            # f.write('\n'.join(map(str, str(lss))))
            f.write(f"tr_loss : {round(lss, 7)} \t ts_loss : {round(test_loss, 7)}\n")

    ## save model and loss
    torch.save(convAE_model.state_dict(), f"./result_model/convAE_model_final({epoch})_0219_1.pth")
    torch.save(loss, f"./result_model/loss_final({epoch}).txt")



    wandb.finish()





    
# test_loss, test_accuracy = evaluate(model, test_loader)
# print(f"\n[EPOCH: {Epoch}]\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy} % \n")
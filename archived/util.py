# import numpy as np
# import torch
# from torch.optim import Adam
# from tqdm import tqdm
# import pickle
# import metrics
# import diffusion as model 



# def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
#     optimizer = Adam(model.parameters(), lr=config["lr"])
#     #ema = EMA(0.9)
#     #ema.register(model)
    
#     if foldername != "":
#         output_path = foldername + "/model.pth"
#         final_path = foldername + "/final.pth"
        
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=150, gamma=.1, verbose=True
#     )
    
#     best_valid_loss = 1e10
    
#     for epoch_no in range(config["epochs"]):
#         avg_loss = 0
#         model.train()
        
#         with tqdm(train_loader) as it:
#             for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
#                 clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
#                 optimizer.zero_grad()
                
#                 loss = model(clean_batch, noisy_batch)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
#                 optimizer.step()
#                 avg_loss += loss.item()
                
#                 #ema.update(model)
                
#                 it.set_postfix(
#                     ordered_dict={
#                         "avg_epoch_loss": avg_loss / batch_no,
#                         "epoch": epoch_no,
#                     },
#                     refresh=True,
#                 )
            
#             lr_scheduler.step()
            
#         if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
#             model.eval()
#             avg_loss_valid = 0
#             with torch.no_grad():
#                 with tqdm(valid_loader) as it:
#                     for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
#                         clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
#                         loss = model(clean_batch, noisy_batch)
#                         avg_loss_valid += loss.item()
#                         it.set_postfix(
#                             ordered_dict={
#                                 "valid_avg_epoch_loss": avg_loss_valid / batch_no,
#                                 "epoch": epoch_no,
#                             },
#                             refresh=True,
#                         )
            
#             if best_valid_loss > avg_loss_valid/batch_no:
#                 best_valid_loss = avg_loss_valid/batch_no
#                 print("\n best loss is updated to ",avg_loss_valid / batch_no,"at", epoch_no,)
                
#                 if foldername != "":
#                     torch.save(model.state_dict(), output_path)
    
#     torch.save(model.state_dict(), final_path)     
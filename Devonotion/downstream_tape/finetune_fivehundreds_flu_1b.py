import os
import time
import scipy.stats
from Devonotion import loadingData, esm, model
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import sys
import numpy as np
from apex import amp
from apex.parallel import convert_syncbn_model
import torch
from torch.utils.data.dataloader import DataLoader
import random
import wandb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2022)

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

wandb.init(project="activeLearning", entity="jessiedraw")
wandb.config = {
    "learning_rate": 1e-5,
    "epochs": 2000,
    "batch_size": 16
}

if __name__ == '__main__':
    epochs = 2000
    batch_size = 16
    fluorescence_train_data = loadingData.FluorescenceDatasetFiveHundreds('train')
    fluorescence_valid_data = loadingData.FluorescenceDatasetFiveHundreds('valid')
    fluorescence_test_data = loadingData.FluorescenceDatasetFiveHundreds('test')

    fluorescence_train_loader = DataLoader(
        fluorescence_train_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_train_data.collaten_fn
    )
    fluorescence_valid_loader = DataLoader(
        fluorescence_valid_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_valid_data.collaten_fn
    )
    fluorescence_test_loader = DataLoader(
        fluorescence_test_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_test_data.collaten_fn
    )

    # model要加
    # model, alphabet = esm.pretrained.load_model_and_alphabet("/research/wzy/esm1b/esm1b_t33_650M_UR50S.pt")
    # model, alphabet = esm.pretrained.load_model_and_alphabet("/home/public/chenlei/protein_finetuning/pretrained_model/esm1b_t33_650M_UR50S.pt")
    model = model.model_down_tape.ProteinBertForValuePrediction().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model = convert_syncbn_model(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    best_loss = 100000
    best_p = 0
    model.train()

    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_p = 0
        train_step = 0
        for idx, batch in enumerate(fluorescence_train_loader):
            fluorescence_inputs = batch['input_ids']
            fluorescence_targets = batch['targets']
            fluorescence_inputs, fluorescence_targets = fluorescence_inputs.cuda(), fluorescence_targets.cuda()
            outputs = model(fluorescence_inputs, targets=fluorescence_targets)
            loss, value_prediction = outputs
            p = spearmanr(fluorescence_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())

            train_loss += loss.item()
            train_p += p
            train_step += 1

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            wandb.log({"train_loss": train_loss / train_step})
            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training Spearman’s: {:.2f}."
                      .format(train_step, len(fluorescence_train_loader), (train_loss / train_step), (train_p / train_step)))
            wandb.watch(model)
        train_toc = time.time()

        model.eval()
        val_tic = time.time()
        val_loss = 0
        val_p = 0
        val_step = 0
        for idx, batch in enumerate(fluorescence_valid_loader):
            fluorescence_inputs = batch['input_ids']
            fluorescence_targets = batch['targets']
            fluorescence_inputs, fluorescence_targets = fluorescence_inputs.cuda(), fluorescence_targets.cuda()
            with torch.no_grad():
                outputs = model(fluorescence_inputs, targets=fluorescence_targets)
                loss, value_prediction = outputs
                p = spearmanr(fluorescence_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())

            val_loss += loss.item()
            val_p += p
            val_step += 1

        # val_step > 0 and val_step % 100 == 0:
        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating Spearman’s: {:.2f}.\n".
                format(val_step, len(fluorescence_valid_loader), (val_loss / val_step), (val_p / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_p = val_p / val_step
        # if val_loss < best_loss:
        if val_p > best_p:
            save_data = {"model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
            print("Save model! Best val Spearman’s is: {:.2f}.".format(val_p))
            torch.save(save_data, "../save/downstream/best_flu_ori.pt")
            best_p = val_p
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))
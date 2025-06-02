import os

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        #加载权重
        if torch.cuda.is_available():
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cuda:0')))
            world.cprint(f"loaded model weights from {weight_file}")
            embedding_user=Recmodel.embedding_user.cpu().weight.detach().numpy()
            embedding_item=Recmodel.embedding_item.cpu().weight.detach().numpy()
        else:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
            embedding_user = Recmodel.embedding_user.weight.detach().numpy()
            embedding_item = Recmodel.embedding_item.weight.detach().numpy()
        emb_file_pth=os.path.join("../data",world.dataset)
        print("emb_file_pth: ",emb_file_pth)
        np.save('emb_user_{}.npy'.format(world.dataset), embedding_user)
        np.save('emb_item_{}.npy'.format(world.dataset), embedding_item)
        exit(0)
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    patience=world.TRAIN_patience
    patience_cnt=0
    hasInit=False
    testSucc=False
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10==0 or epoch==1:
            cprint("[TEST]")
            try:
                results=Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                testSucc=True
            except:
                testSucc=False

            if epoch==0 and testSucc:
                metric_best=results
                need_stop=True
                hasInit=True
            elif epoch==1 and not hasInit and testSucc:
                metric_best = results
                need_stop = True
                hasInit = True
            elif testSucc:
                metric_best,need_stop=Procedure.early_stopping(metric_best,results)
                if need_stop:patience_cnt+=1
                else:#bigger_performance
                    patience_cnt=0
                if patience_cnt>patience:
                    #early stop
                    break
            if testSucc:
                print("need_stop= ",need_stop)
                print('best metrics={} '.format(metric_best))
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        if not need_stop or patience>999:
            torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()

exit(0)

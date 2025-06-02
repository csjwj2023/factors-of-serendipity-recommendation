import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['lastfm', 'gowalla', 'yelp2018', 'amazon-book',"ser_bk","ser_mv","mlls","tool","beauty","clothing","electronics",
                     "home","kindle","sport","ml10m","cloth_5core_onlyRate","sport_5core_onlyRate","kindle_5core_onlyRate"
    ,"elec_5core_onlyRate","home_5core_onlyRate"]:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
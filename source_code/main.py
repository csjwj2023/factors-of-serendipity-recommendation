import recommend
import utils

if __name__ == "__main__":
    list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]#[7:8]
    list_method = ["rand", "nov", "pop", "qua", "ela", "acc", "dif", "div"]
    list_dataset_name = list_dataset_name#[5:]
    K = 20
    list_seed = [777, 7777, 77777, 73, 79, 83, 89, 97, 101, 103][:5]  # [0:1]
    recommend.recommend(list_dataset_name, list_seed, K)
    # utils.evaluate(list_dataset_name, list_seed, list_method, K)

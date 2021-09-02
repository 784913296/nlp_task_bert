# coding=utf-8
""" 将数据集文件切成 train dev test """
from sklearn.model_selection import train_test_split

def write_data(data, file_name, save_path):
    with open(save_path + file_name, 'w', encoding="utf-8") as f:
        for line in data:
            f.write(line)


def data_split(data_file, save_path):
    print("正在生成数据集文件")
    lines_list = []
    with open(data_file, 'r', encoding="utf-8") as f:
        for i in range(93000):
            line = f.readline().replace(' ', '')
            lines_list.append(line)
    train, test = train_test_split(lines_list, train_size=0.9, shuffle=True)
    train, dev = train_test_split(train, train_size=0.9, shuffle=True)
    write_data(train, 'train.txt', save_path)
    write_data(dev, 'dev.txt', save_path)
    write_data(test, 'test.txt', save_path)
    return "数据集 test.txt dev.txt test.txt 已生成"




# data_file = "task_sim/data/train_data/all_title_query_pair.txt"
# print(data_split(data_file))

# data_file = "../task_classes/data/train_data"
# save_path = "../task_classes/data/"
# data_split(data_file, save_path)
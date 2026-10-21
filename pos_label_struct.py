import json
import spacy
import pickle
from tqdm import tqdm
# 读取jsonl文件并提取指定字段
# 这里是直接读取数据信息，然后进行词性解析
'''
def read_jsonl_text(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append({
                'question_id': json_obj['question_id'],
                'text': json_obj['text']
            })
    return data


def read_jsonl_image(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            # 去掉图片的文件扩展名（.jpg）
            image_name = json_obj['image'].split('.')[0]  # 这里用split去掉.jpg
            data.append({
                'question_id': json_obj['question_id'],
                'id': image_name
            })
    return data


# 合并两个列表
def merge_by_question_id(text_data, image_data):
    merged_data = {}

    # 将 text_data 以 question_id 为键存入字典
    for item in text_data:
        question_id = item['question_id']
        merged_data[question_id] = {'question_id': question_id, 'text': item['text']}

    # 将 image_data 以 question_id 为键加入字典
    for item in image_data:
        question_id = item['question_id']
        if question_id in merged_data:
            merged_data[question_id]['id'] = item['id']
        else:
            merged_data[question_id] = {'question_id': question_id, 'id': item['id']}

    # 将字典转换回列表形式
    merged_list = list(merged_data.values())
    return merged_list

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

# 读取文件数据
text_file_path = '/home/nlp306/Data/User_file/wgj/Mypro/DynRT-main/train/answers_hf.jsonl'  # 替换为 text 文件路径
image_file_path = '/home/nlp306/Data/User_file/wgj/Mypro/DynRT-main/train/train_questions.jsonl'  # 替换为 image 文件路径

text_data = read_jsonl_text(text_file_path)
image_data = read_jsonl_image(image_file_path)

# 合并数据
merged_data = merge_by_question_id(text_data, image_data)

train_text = load_file('input/prepared_clean/train_text')
train_id = load_file('input/prepared_clean/train_id')

# 合并函数
def merge_lists(texts, ids):
    merged_data = []
    for text, id_ in zip(texts, ids):
        merged_data.append({
            'text': text,
            'id': id_
        })
    return merged_data

# 调用合并函数
merged_result = merge_lists(train_text, train_id)

# 合并两个列表的函数
def merge_by_id(list1, list2):
    merged_temp = []

    # 创建一个字典，以 list2 的 id 为键，存储对应数据
    dict_list2 = {item['id']: item for item in list2}

    # 遍历 list1，根据 id 合并数据
    for item1 in list1:
        id_ = item1['id']
        if id_ in dict_list2:
            merged_item = {
                'id': id_,
                'text_list1': item1['text'],
                'text_list2': dict_list2[id_]['text'],
                # 'question_id': dict_list2[id_]['question_id']
            }
            merged_temp.append(merged_item)

    return merged_temp

# 合并两个列表
merged_final = merge_by_id(merged_data, merged_result)

'''

nlp = spacy.load('en_core_web_sm')

# 定义计算相似度并提取词性集合的函数
def process_texts(text_list1, text_list2):
    # 将文本转为 spaCy 文档对象
    doc1 = nlp(text_list1)
    doc2 = nlp(text_list2)

    # 计算 text_list2 中单词和 text_list1 中单词的相似度，并找到前十个最相似的单词
    similarities = []

    for token2 in doc2:
        if not token2.is_stop and token2.is_alpha:  # 忽略停用词和非字母符号
            for token1 in doc1:
                if not token1.is_stop and token1.is_alpha:
                    sim = token1.similarity(token2)
                    similarities.append((token1.text, token2.text, sim))

    # 按照相似度排序，获取相似度最高的前10个
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:10]

    # 获取 text_list2 中最相似单词的词性，并构建词性集合（确保唯一性）
    pos_tags = []
    pos_set = set()  # 存储唯一的词性

    for token1, token2, sim in similarities:
        token_doc = nlp(token2)  # 这里获取的是 text_list2 中的词性
        for token in token_doc:
            pos_tags.append({
                'word': token.text,
                'POS': token.pos_,
                'lemma': token.lemma_,
                'similarity': sim
            })
            pos_set.add(token.pos_)  # 将词性加入集合（保证唯一性）

    return pos_tags, pos_set

# 存储所有数据中的词性
global_pos_set = set()

# 更新词性辅助预测任务的词性
# for item in tqdm(merged_final):
#     # print(f"\nProcessing ID: {item['id']}")
#     text_list1 = item['text_list1']
#     text_list2 = item['text_list2']
#
#     pos_tags, pos_set = process_texts(text_list1, text_list2)
#     # 合并当前数据中的词性到全局集合中
#     global_pos_set.update(pos_set)

#global_pos_set = {'ADJ', 'PRON', 'NOUN', 'PROPN', 'VERB', 'NUM', 'PART', 'INTJ', 'ADV', 'CCONJ', 'SCONJ'}
print(global_pos_set)
# {'ADJ', 'PRON', 'NOUN', 'PROPN', 'X', 'VERB', 'NUM', 'PART', 'INTJ', 'ADV', 'ADP', 'CCONJ', 'SCONJ'}，将数据进行解析，然后将这些词性的数据构造成数据标签，其他的数据置为other，然后词性辅助任务预测。
# 根据词性标签，重新处理每一个句子，将句子中的单词进行词性标注，然后做成数据集用于词性标签预测
# 对每个句子进行词性标注，并保存为pkl文件
# 训练集中的词性{'ADP', 'NOUN', 'X', 'ADJ', 'SCONJ', 'ADV', 'NUM', 'PART', 'PRON', 'PROPN', 'CCONJ', 'INTJ', 'VERB'} 多了X, ADP是介词
# global_pos_set = {'NOUN', 'ADJ', 'SCONJ', 'ADV', 'NUM', 'PART', 'PRON', 'PROPN', 'CCONJ', 'INTJ', 'VERB'}
global_pos_set = {'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM', 'ADV', 'INTJ'}
def process_and_save_pkl(data, output_pkl):
    all_results = []

    for item in data:
        text_list2 = item['text_list2']
        doc = nlp(text_list2)
        sentence_results = []

        # 记录每个单词及其词性，并根据 global_pos_set 进行分类
        for i, token in enumerate(doc):
            pos_tag = token.pos_ if token.pos_ in global_pos_set else 'OTHER'
            sentence_results.append({
                'index': i,
                'word': token.text,
                'pos': pos_tag
            })

        # 将每个句子的结果添加到 all_results 列表中
        all_results.append({
            'id': item['id'],
            'sentence': text_list2,
            'tokens': sentence_results
        })

    # 保存为pkl文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for result in all_results:
    #         f.write(f"ID: {result['id']}\n")
    #         f.write(f"Sentence: {result['sentence']}\n")
    #         f.write("Tokens:\n")
    #         for token in result['tokens']:
    #             f.write(f"  Index: {token['index']}, Word: {token['word']}, POS: {token['pos']}\n")
    #         f.write("\n")  # 每条记录之间空一行


# 输出文件路径
output_file = '/home/nlp306/Data/User_file/wgj/Mypro/DynRT-main/input/prepared_clean/train_pos.pkl'  # 替换为你想要输出的文件路径

# 处理数据并输出到文件
# process_and_save_pkl(merged_final, output_file)

# 训练过程中就是读取数据，然后进行特征预测，将文本特征预测词性之后与嵌入特征相融合。

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

valid_text = load_file('input/prepared/valid_text')
valid_id = load_file('input/prepared/valid_id')
test_text = load_file('input/prepared/test_text')
test_id = load_file('input/prepared/test_id')

def merge_lists(texts, ids):
    merged_data = []
    for text, id_ in zip(texts, ids):
        merged_data.append({
            'text': text,
            'id': id_
        })
    return merged_data

valid_merged_result = merge_lists(valid_text, valid_id)
test_merged_result = merge_lists(test_text, test_id)
def transform_data(data):
    transformed_data = []
    for item in data:
        transformed_item = {
            'id': item['id'],
            'text_list1': "",  # 增加 'text_list1'，内容为空
            'text_list2': item['text']  # 将 'text' 内容赋给 'text_list2'
        }
        transformed_data.append(transformed_item)
    return transformed_data

valid_merged = transform_data(valid_merged_result)
test_merged = transform_data(test_merged_result)
process_and_save_pkl(valid_merged, './input/prepared/valid_pos.pkl')
process_and_save_pkl(test_merged, './input/prepared/test_pos.pkl')

import pickle

# 替换为你的 .pkl 文件路径
file_path = './input/prepared_clean/test_text'

# 读取 .pkl 文件
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 打印读取的数据
# print(data)

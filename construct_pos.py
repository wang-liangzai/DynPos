# -*- coding: utf-8 -*-
import pickle
import spacy
from transformers import RobertaTokenizer


def load_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_file(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def map_pos(pos):
    keep_pos = {
        'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM', 'ADV', 'INTJ'
    }
    if pos in keep_pos:
        return pos
    return "other"


if __name__ == "__main__":

    # ========= 路径 =========
    data_path = "input/prepared_clean/"

    test_id_path = data_path + "valid_id"
    test_text_path = data_path + "valid_text"

    save_path = data_path + "valid_pos"

    # 本地 roberta 路径
    roberta_path = "./roberta-base"

    # ========= 加载 =========
    test_ids = load_file(test_id_path)
    test_texts = load_file(test_text_path)

    assert len(test_ids) == len(test_texts), "test_id 和 test_text 数量不一致"

    nlp = spacy.load("en_core_web_sm")
    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)

    result = []

    for sample_id, sentence in zip(test_ids, test_texts):

        doc = nlp(sentence)

        tokens = []
        index = 0

        for token in doc:
            word = token.text
            pos = map_pos(token.pos_)

            # RoBERTa 可能会把一个词切成多个子词
            sub_tokens = tokenizer.tokenize(word)

            # 如果没有切出来，就保留原词
            if len(sub_tokens) == 0:
                tokens.append({
                    "index": index,
                    "word": word,
                    "pos": pos
                })
                index += 1
            else:
                for sub in sub_tokens:
                    clean_sub = sub.replace("Ġ", "")

                    if clean_sub == "":
                        clean_sub = word

                    tokens.append({
                        "index": index,
                        "word": clean_sub,
                        "pos": pos
                    })
                    index += 1

        item = {
            "id": sample_id,
            "sentence": sentence,
            "tokens": tokens
        }

        result.append(item)

    save_file(result, save_path)

    print("转换完成！")
    print("保存路径：", save_path)
    print("样本数量：", len(result))
    print("第一条样本：")
    print(result[0])
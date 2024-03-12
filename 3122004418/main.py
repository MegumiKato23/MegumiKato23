import re
import sys
import jieba
# 计算词频
from collections import Counter
import numpy as np

def read_file(file_path):
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def jieba_tokenize(text):
    # 使用 jieba 的全模式进行分词
    return list(jieba.cut(text, cut_all=True))

def del_nothing(text):
    # 使用正则表达式删除符号
    return re.sub(r'[\n\s\.,.，。、“”:：;!?()"\-]', "", text)

def word_frequency(text):
    # 计算每个词的词频
    word_counts = Counter(text)

    max_values = max(word_counts.values())

    for word, count in word_counts.items():
        word_counts[word] = count / max_values

    return word_counts
    
def word_composition(counter1, counter2):
    # 创建一个新的 Counter 对象来保存结果
    result = Counter()

    # 遍历第一个 Counter 对象
    for key in counter1:
        # 如果键在第二个 Counter 对象中也存在，则保留两个值
        if key in counter2:
            result[key] = (counter1[key], counter2[key])
        else:
            # 如果键只存在于第一个 Counter 对象中，则只保留第一个值
            result[key] = (counter1[key], 0)

    # 遍历第二个 Counter 对象
    for key in counter2:
        # 如果键只存在于第二个 Counter 对象中，则只保留第二个值
        if key not in counter1:
            result[key] = (0, counter2[key])

    return result

def calculate_cosine_similarity(counter):
    original_list = list(counter.values())

    list_x = [x for x, y in original_list]
    list_y = [y for x, y in original_list]

    # 计算向量的欧几里得范数
    norm_x = np.linalg.norm(list_x)
    norm_y = np.linalg.norm(list_y)

    return np.dot(list_x, list_y) / (norm_x * norm_y)
    
def calculate_similarity_tf(original_text, plagiarized_text):
    # 分词
    original_tokens = jieba_tokenize(del_nothing(original_text))
    plagiarized_tokens = jieba_tokenize(del_nothing(plagiarized_text))
    
    # 计算词频
    original_counter = word_frequency(original_tokens)
    plagiarized_counter = word_frequency(plagiarized_tokens)

    # 计算总词集 
    re_counter = word_composition(original_counter, plagiarized_counter)

    return calculate_cosine_similarity(re_counter)

def main():
    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 读取文件内容
    original_text = read_file(sys.argv[1])
    plagiarized_text = read_file(sys.argv[2])
    
    # 计算相似度
    similarity = calculate_similarity_tf(original_text, plagiarized_text)

    print(similarity)
    
    # 输出结果到文件
    with open(sys.argv[3], 'w', encoding='utf-8') as output_file:
        output_file.write(f"Similarity: {similarity:.2%}")

if __name__ == '__main__':
    # 确保命令行参数数量正确
    if len(sys.argv) != 4:
        print("使用方法:python main.py <原文路径> <抄袭文路径> <输出路径>")
        sys.exit(1)
    else:
        main()
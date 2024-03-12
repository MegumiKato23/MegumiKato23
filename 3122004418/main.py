# 导入库
import re
import sys
import jieba
from collections import Counter
import numpy as np

def read_file(file_path):
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def detect_language(text):
    # 对文档的语言进行区分
    chinese_count = len(re.findall(u'[\u4e00-\u9fff]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    
    if chinese_count > english_count:
        return "Chinese"
    elif english_count > chinese_count:
        return "English"

def jieba_tokenize(text):
    # 使用 jieba 的精确模式进行分词
    return list(jieba.cut(text))

def english_tokenize(text):
    # 对英文文章进行分词
    return text.split(' ')

def del_punct(text):
    # 使用正则表达式删除符号
    return re.sub(r'[\n\s\.,.，。、’“”:：;!！?？()（）"\'\-]', "", text)

def del_punct_eng(text):
    # 使用正则表达式替换英文的标点符号为空格
    return re.sub(r'[^\w\s]', '', text)

# 计算每个词的词频，并返回一个Counter类型
def word_frequency(text):
    if text == [""]:
        return Counter()
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

# 计算余弦相似度
def calculate_cosine_similarity(counter):
    original_list = list(counter.values())

    list_x = [x for x, _ in original_list]
    list_y = [y for _, y in original_list]

    # 计算向量的欧几里得范数
    norm_x = np.linalg.norm(list_x)
    norm_y = np.linalg.norm(list_y)

    return np.dot(list_x, list_y) / (norm_x * norm_y)
    
def calculate_similarity_tf(original_text, plagiarized_text):
    # 分词
    if detect_language(original_text) == "Chinese":
        original_tokens = jieba_tokenize(del_punct(original_text))
        plagiarized_tokens = jieba_tokenize(del_punct(plagiarized_text))
    else:
        original_tokens = english_tokenize(del_punct_eng(original_text))
        plagiarized_tokens = english_tokenize(del_punct_eng(plagiarized_text))
    
    # 计算词频
    original_counter = word_frequency(original_tokens)
    plagiarized_counter = word_frequency(plagiarized_tokens)

    # 计算总词集 
    re_counter = word_composition(original_counter, plagiarized_counter)

    return calculate_cosine_similarity(re_counter)

def main():
    # 确保命令行参数数量正确
    if len(sys.argv) != 4:
        print("使用方法:python main.py <原文路径> <抄袭文路径> <输出路径>")
        sys.exit(1)

    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 读取文件内容
    original_text = read_file(original_path)
    plagiarized_text = read_file(plagiarized_path)
    
    # 计算相似度
    similarity = calculate_similarity_tf(original_text, plagiarized_text)
    
    # 输出结果到文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"Similarity: {similarity:.2%}")

if __name__ == '__main__':
    main()
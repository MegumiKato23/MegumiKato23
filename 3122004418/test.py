import unittest
import os
import tempfile
from unittest.mock import patch
from main import *

class Test(unittest.TestCase):

    def test_read_file(self):
        # 创建一个临时文件并写入一些文本
        content = "这是一个测试文件的内容。"
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as file:
            file.write(content)
            file_path = file.name

        # 调用函数并检查返回内容
        read_content = read_file(file_path)
        self.assertEqual(content, read_content)

        # 删除临时文件
        os.remove(file_path)

    def test_detect_language(self):
        # 创建一个临时文件并写入一些文本
        content = "这是一个测试文件的内容。"
        language = detect_language(content)
        self.assertEqual("Chinese", language)

        content = "This is the content of a test file."
        language = detect_language(content)
        self.assertEqual("English", language)

    def test_english_tokenize(self):
        text = "This is a test sentence."
        expected_tokens = ["This", "is", "a", "test", "sentence."]
        tokens = english_tokenize(text)
        self.assertEqual(tokens, expected_tokens)


class TestJiebaTokenize(unittest.TestCase):
    def test_jieba_tokenize(self):
        # 测试简单的文本
        text = "我爱北京天安门"
        expected_tokens = ["我", "爱", "北京", "天安门"]
        tokens = jieba_tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_jieba_tokenize_empty_string(self):
        # 测试空字符串
        text = ""
        expected_tokens = []
        tokens = jieba_tokenize(text)
        self.assertEqual(tokens, expected_tokens)

class TestDelPunct(unittest.TestCase):
    def test_del_punct(self):
        # 测试简单的文本
        text = "Hello, World!"
        expected_result = "HelloWorld"
        result = del_punct(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_with_chinese(self):
        # 测试包含中文标点符号的文本
        text = "你好，世界！"
        expected_result = "你好世界"
        result = del_punct(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_with_newlines(self):
        # 测试包含换行符的文本
        text = "Hello\nWorld"
        expected_result = "HelloWorld"
        result = del_punct(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_with_whitespace(self):
        # 测试包含空格的文本
        text = "Hello World"
        expected_result = "HelloWorld"
        result = del_punct(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_with_mixed_punct(self):
        # 测试包含多种标点符号的文本
        text = "Hello, World! How's it going?"
        expected_result = "HelloWorldHowsitgoing"
        result = del_punct(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_empty_string(self):
        # 测试空字符串
        text = ""
        expected_result = ""
        result = del_punct(text)
        self.assertEqual(result, expected_result)

class TestDelPunctEng(unittest.TestCase):
    def test_del_punct_eng(self):
        # 测试简单的文本
        text = "Hello, World!"
        expected_result = "Hello World"
        result = del_punct_eng(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_eng_with_multiple_punct(self):
        # 测试包含多种英文标点符号的文本
        text = "Hello, World! How's it going?"
        expected_result = "Hello World Hows it going"
        result = del_punct_eng(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_eng_with_numbers(self):
        # 测试包含数字的文本
        text = "123, 456!"
        expected_result = "123 456"
        result = del_punct_eng(text)
        self.assertEqual(result, expected_result)

    def test_del_punct_eng_with_whitespace(self):
        # 测试包含空格的文本
        text = "Hello,  World! "
        expected_result = "Hello  World "
        result = del_punct_eng(text)
        self.assertEqual(result, expected_result)
        
    def test_del_punct_eng_empty_string(self):
        # 测试空字符串
        text = ""
        expected_result = ""
        result = del_punct_eng(text)
        self.assertEqual(result, expected_result)

class TestWordFrequency(unittest.TestCase):
    def test_word_frequency(self):
        # 测试简单的文本
        text = "hello hello world world"
        expected_result = {'hello': 1.0, 'world': 1.0}
        result = word_frequency(english_tokenize(text))
        self.assertEqual(result, expected_result)

    def test_word_frequency_with_different_counts(self):
        # 测试词频不同的文本
        text = "hello world hello hello"
        expected_result = {'hello': 1.0, 'world': 0.3333333333333333}
        result = word_frequency(english_tokenize(text))
        self.assertEqual(result, expected_result)

    def test_word_frequency_with_single_word(self):
        # 测试只有一个词的文本
        text = "hello"
        expected_result = {'hello': 1.0}
        result = word_frequency(english_tokenize(text))
        self.assertEqual(result, expected_result)

    def test_word_frequency_with_empty_string(self):
        # 测试空字符串
        text = ""
        expected_result = Counter()
        result = word_frequency(english_tokenize(text))
        self.assertEqual(result, expected_result)

    def test_word_frequency_with_non_alphabetic_text(self):
        # 测试非字母文本
        text = "123 456"
        expected_result = {'123': 1.0, '456': 1.0}
        result = word_frequency(english_tokenize(text))
        self.assertEqual(result, expected_result)

class TestWordComposition(unittest.TestCase):
    def test_word_composition_equal_counters(self):
        # 测试两个相同的`Counter`对象
        counter1 = Counter({'a': 1, 'b': 2, 'c': 3})
        counter2 = Counter({'a': 1, 'b': 2, 'c': 3})
        expected_result = Counter({'a': (1, 1), 'b': (2, 2), 'c': (3, 3)})
        result = word_composition(counter1, counter2)
        self.assertEqual(result, expected_result)

    def test_word_composition_one_extra_in_counter1(self):
        # 测试`Counter1`有一个额外的键
        counter1 = Counter({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        counter2 = Counter({'a': 1, 'b': 2, 'c': 3})
        expected_result = Counter({'a': (1, 1), 'b': (2, 2), 'c': (3, 3), 'd': (4, 0)})
        result = word_composition(counter1, counter2)
        self.assertEqual(result, expected_result)

    def test_word_composition_one_extra_in_counter2(self):
        # 测试`Counter2`有一个额外的键
        counter1 = Counter({'a': 1, 'b': 2, 'c': 3})
        counter2 = Counter({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        expected_result = Counter({'a': (1, 1), 'b': (2, 2), 'c': (3, 3), 'd': (0, 4)})
        result = word_composition(counter1, counter2)
        self.assertEqual(result, expected_result)

    def test_word_composition_empty_counters(self):
        # 测试两个空的`Counter`对象
        counter1 = Counter()
        counter2 = Counter()
        expected_result = Counter()
        result = word_composition(counter1, counter2)
        self.assertEqual(result, expected_result)

class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity_simple(self):
        # 测试简单的向量
        counter = Counter({'a': (1, 2), 'b': (3, 2), 'c': (3, 4)})
        expected_result = 0.936585811581694
        result = calculate_cosine_similarity(counter)
        self.assertAlmostEqual(result, expected_result)

    def test_cosine_similarity_same_values(self):
        # 测试向量中所有值都相等的情况
        counter = Counter({'a': (5, 1), 'b': (5, 1), 'c': (5, 1)})
        expected_result = 1.0
        result = calculate_cosine_similarity(counter)
        self.assertAlmostEqual(result, expected_result)

    def test_cosine_similarity_orthogonal_vectors(self):
        # 测试正交向量（余弦相似度为0）
        counter = Counter({'a': (1, 0), 'b': (0, 1), 'c': (0, 1)})
        expected_result = 0.0
        result = calculate_cosine_similarity(counter)
        self.assertAlmostEqual(result, expected_result)

class TestCalculateSimilarityTF(unittest.TestCase):
    def test_calculate_similarity_tf_chinese_texts(self):
        # 测试中文文本
        original_text = "这是一个测试句子。"
        plagiarized_text = "这是一个测试句子。"
        expected_result = 1.0  # 假设两个中文句子完全相同
        result = calculate_similarity_tf(original_text, plagiarized_text)
        self.assertAlmostEqual(result, expected_result)

    def test_calculate_similarity_tf_english_texts(self):
        # 测试英文文本
        original_text = "This is a test sentence. This is a test sentence."
        plagiarized_text = "This is a test sentence. This is a test sentence."
        expected_result = 1.0  # 假设两个英文句子完全相同
        result = calculate_similarity_tf(original_text, plagiarized_text)
        self.assertAlmostEqual(result, expected_result)

class TestMainWithArgs(unittest.TestCase):
    @patch('sys.argv', ['script.py', 'original.txt', 'plagiarized.txt', 'output.txt'])
    def test_main_with_args(self):
        with patch('sys.argv', new=['main.py', 'orig.txt', 'orig_0.8_add.txt', 'output.txt']):
            main()
            self.assertEqual(sys.argv[1], 'orig.txt')
            self.assertEqual(sys.argv[2], 'orig_0.8_add.txt')
            self.assertEqual(sys.argv[3], 'output.txt')

    def test_main_without_args(self):
        with patch('sys.argv', new=['main.py']):
             with self.assertRaises(SystemExit):
                main()

if __name__ == '__main__':
    unittest.main()
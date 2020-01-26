import numpy as np
def get_dataset(data_path, num_samples):
    input_texts = []
    target_texts = []

    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, _ = line.split('\t')
        # 用tab作用序列的开始，用\n作为序列的结束
        target_text = '\t' + target_text + '\n'

        input_texts.append(input_text)
        target_texts.append(target_text)
        
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    return input_texts,target_texts,input_characters,target_characters


# 一共10000个样本
num_samples = 10000

# 读取数据集
data_path = 'fra.txt'

# 获取数据集
# 其中input_texts为输入的英文字符串
# target_texts为对应的法文字符串

# input_characters用到的所有输入字符,如a,b,c,d,e,……,.,!等
# target_characters用到的所有输出字符
input_texts,target_texts,input_characters,target_characters = get_dataset(data_path, num_samples)

# 对字符进行排序
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# 计算共用到了什么字符
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# 计算出最长的序列是多长
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('一共有多少训练样本：', len(input_texts))
print('多少个英文字母：', num_encoder_tokens)
print('多少个法文字母：', num_decoder_tokens)
print('最大英文序列:', max_encoder_seq_length)
print('最大法文序列:', max_decoder_seq_length)

# 建立字母到数字的映射
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

#---------------------------------------------------------------------------#

#--------------------------------------#
#   改变数据集的格式
#--------------------------------------#
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
    
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # 为末尾加上" "空格
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    
    # 相当于前一个内容的识别结果，作为输入，传入到解码网络中
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data不包括第一个tab
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
#---------------------------------------------------------------------------#
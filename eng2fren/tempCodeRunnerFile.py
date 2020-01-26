
# # 建立字母到数字的映射
# input_token_index = dict(
#     [(char, i) for i, char in enumerate(input_characters)])
# target_token_index = dict(
#     [(char, i) for i, char in enumerate(target_characters)])

# #---------------------------------------------------------------------------#

# #--------------------------------------#
# #   改变数据集的格式
# #--------------------------------------#
# encoder_input_data = np.zeros(
#     (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
#     dtype='float32')
# decoder_input_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#     dtype='float32')
# decoder_target_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#     dtype='float32')


# for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#     # 为末尾加上" "空格
#     for t, char in enumerate(input_text):
#         encoder_input_data[i, t, input_token_index[char]] = 1.
#     encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
#     # 相当于前一个内容的识别结果，作为输入，传入到解码网络中
#     for t, char in enumerate(target_text):
#         decoder_input_data[i, t, target_token_index[char]] = 1.
#         if t > 0:
#             # decoder_target_data不包括第一个tab
#             decoder_target_data[i, t - 1, target_token_index[char]] = 1.
#     decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
#     decoder_target_data[i, t:, target_token_index[' ']] = 1.
# #---------------------------------------------------------------------------#
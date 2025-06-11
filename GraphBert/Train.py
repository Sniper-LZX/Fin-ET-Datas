import argparse
import time
import random
import modelOpt.Opt
from transformers import BertTokenizer
from torch.optim import AdamW
from modelOpt.Utils import *


""" -------------- 命令行 + 参数 -------------- """
parser = argparse.ArgumentParser(
    description='Train.py',  # 只是在命令行参数出现错误的时候，随着错误信息打印出来
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 自定义帮助文档输出格式
# 给命令加入参数，即设置 Train.py参数
modelOpt.Opt.model_opts(parser)
# 解析命令行参数得到对象
opt = parser.parse_args()
opt.bert_model = "chinese-bert-wwm"
opt.max_seq_length = 512

# 通过 parse_gpuid 函数解析得到的 GPU ID 列表
gpu_ls = parse_gpuid(opt.gpuls)
opt.train_batch_size = 32 * len(gpu_ls)

# 设置随机种子
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


if opt.output_dir:
    # 创建模型检查点的目录
    os.makedirs(opt.output_dir, exist_ok=True)

# 加载预训练的 BERT 分词器
print("加载BERT分词器...")
tokenizer = BertTokenizer.from_pretrained(opt.bert_model)

print("加载训练数据...", end="")
train_examples = load_examples("Construct_Data/train_data_235权重.pkl")
train_features = convert_examples_to_features(train_examples, tokenizer, opt.max_seq_length)
print(len(train_features))

print("加载测试数据...", end="")
test_examples_all = load_examples("Construct_Data/test_data_235权重.pkl")
test_features_all = convert_examples_to_features(test_examples_all, tokenizer, opt.max_seq_length)
print(len(test_features_all))

""" 这行代码计算训练过程中的总步骤数 """
num_train_steps = \
    int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs) * 5


""" ===== 模型构建 GraphBert  ===== """
model = ini_from_pretrained(opt)

model_config = model.config

param_optimizer = list(model.named_parameters())

# 设置神经网络的优化器参数
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': opt.l2_reg},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
# 学习率从一个非常小的值开始，在预热阶段逐步增加到预定的学习率
optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)

model.cuda(gpu_ls[0])
# 使用 NVIDIA 的 Apex 库进行混合精度训练
model = model.to('cuda')
# 使得模型可以在多个GPU上运行
model = torch.nn.DataParallel(model,  device_ids=gpu_ls)
model.config = model_config

global_step = 0

print("***** Running training *****")
print("  Num examples = %d", len(train_examples))
print("  Batch size = %d", opt.train_batch_size)
print("  Num steps = %d", num_train_steps)

# 按批次加载数据集的加载器
train_dataloader = load_data(train_features, batch_size=8)

# 交叉熵损失函数
loss_nsp_fn = torch.nn.CrossEntropyLoss()
''' 损失项权重：平衡不同损失项对总损失的贡献 '''
Lambda = opt.Lambda
best_step = 0

# 根据opt对象中的不同属性和选项，生成一个描述这些选项的字符串名称。
time_start = str(int(time.time()))[-6:]
temp_data = {}
# 训练循环
for epoch in range(int(opt.num_train_epochs)):
    print("Epoch:", epoch)
    # 累积训练损失
    tr_loss = 0
    # 分别记录训练样本数和训练步数
    nb_tr_examples, nb_tr_steps = 0, 0
    accuracy = 0
    if True:
        # 按批次加载数据 step:当前的训练步骤  batch:当前批次的数据，包括输入特征和标签。
        for step, batch in enumerate(train_dataloader):
            model.train()
            # 将批次中的每一个张量（tensor）都转移到指定的 GPU 上
            batch = tuple(t.cuda(gpu_ls[0]) for t in batch)

            example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = batch
            # 获取选项数
            num_choices = input_ids.shape[1]
            if (step * opt.train_batch_size) % 100 == 0:
                # 从所有测试特征中随机选择300个进行评估
                # test_features = random.sample(test_features_all, 300)
                test_dataloader = load_data(test_features_all, batch_size=opt.train_batch_size, dat_type='test')
                model = model.eval()
                # 对模型进行评估，并返回准确性（保存预测分数）
                accuracy, cls_score_tmp, answers_tmp = do_evaluation(model, test_dataloader, opt, gpu_ls)
                # 预测分数
                temp_data['cls_score'] = cls_score_tmp.tolist()
                temp_data['answers'] = answers_tmp.tolist()

                print('step:', step, "accuracy:", "{:.4f}".format(accuracy))
                # 将模型设置为训练模型
                model = model.train()

            # for循环遍历num_choices次，即选项个数
            for n in range(num_choices):
                input_ids_tmp = input_ids[:, n, :]
                input_masks_tmp = input_masks[:, n, :]
                segment_ids_tmp = segment_ids[:, n, :]
                sentence_inds_tmp = sentence_inds[:, n, :]
                graphs_tmp = graphs[:, n, :]
                answers_tmp = answers[:, n]

                graphs_tmp_scaled = graphs_tmp

                # 计算类别分数（cls_scores）、注意力分数（attn_scores）和图形数据（graphs_tmp_scaled）
                _, cls_scores, attn_scores,  = model(input_ids=input_ids_tmp,
                                                     token_type_ids=segment_ids_tmp,
                                                     sentence_inds=sentence_inds_tmp,
                                                     graphs=graphs_tmp_scaled)

                # 计算了类别分数（cls_scores）和正确答案（answers_tmp）之间的损失
                loss = loss_nsp_fn(cls_scores, answers_tmp)

                if step % 20 == 0:
                    print("step:", step, "loss_nsp:", loss.detach().cpu().numpy())

                # 记录损失
                # f = open('../loss_information/' + epoch + '_' + step + '_' + time_start + '.csv', 'w')
                # f.write(str(loss.detach().cpu().numpy()) + '\n')
                # f.close()

                loss.backward()
                # 这里将当前计算出的损失加到总损失 tr_loss 中，loss.item() 把损失从一个tensor转换成了一个Python数值。
                tr_loss += loss.item()
                # 给出了当前批次中的样本数量，将此数量累加到 nb_tr_examples 变量中
                nb_tr_examples += input_ids.size(0)
                # 跟踪训练步骤的数量，并对其进行递增
                nb_tr_steps += 1
                # 梯度累积和参数更新
                if (step + 1) % opt.gradient_accumulation_steps == 0:
                    # 更新参数
                    optimizer.step()
                    # 梯度清零
                    model.zero_grad()
                    # 全局步数更新
                    global_step += 1
            ls = [model.config, model.state_dict()]

            if epoch == int(opt.num_train_epochs) - 1 and step == int(opt.num_train_epochs) - 1:
                file_name = "预测矩阵/{.4f}.json".format(accuracy)
                prediction_score = open(file_name, 'w', encoding='UTF-8')
                json.dump(temp_data, prediction_score, ensure_ascii=False, indent=4)
                torch.save(ls, "models/FinET_" + str(epoch) + "_" + str(accuracy) + '.pkl')

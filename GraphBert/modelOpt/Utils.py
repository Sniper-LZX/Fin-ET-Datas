import pickle
import numpy as np
import pandas as pd
from .DataSet import *
from .BertModules import *
from .GraphBert import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def load_examples(input_file):
    f = open(input_file, 'rb')
    examples = pickle.load(f)
    f.close()
    return examples
    

def parse_gpuid(gpuls):
    ls = [int(n) for n in str(gpuls)]
    return ls
    
    
def parse_opt_to_name(opt):
    if 'base' in opt.bert_model:
        model_size = 'b'
    else:
        model_size = 'l'
    if opt.use_bert:
        model_type = "b"
    else:
        model_type = 'gb'
    if opt.pretrain:
        pretrain = 'p'
    else:
        pretrain = 'up'
    if opt.link_predict:
        graph_type = 'k'
    else:
        graph_type = 'e'
    start_layer = opt.start_layer
    merge_layer = opt.merge_layer
    n_layer_extractor = opt.n_layer_extractor
    n_layer_aa = opt.n_layer_aa
    n_layer_gnn = opt.n_layer_gnn
    n_layer_merger = opt.n_layer_merger
    method_extractor = opt.method_extractor[0]
    method_merger = opt.method_merger[0]
    smooth_term = opt.loss_aa_smooth
    smooth_method = opt.loss_aa_smooth_method[0]
    lr = opt.learning_rate
    warmup_proportion = opt.warmup_proportion
    margin = opt.do_margin_loss * opt.margin
    Lambda = opt.Lambda
    sep_sent = str(opt.sep_sent)[0]
    layer_norm = str(opt.layer_norm)[0]

    name = [str(n) for n in [pretrain, graph_type, model_type, model_size, start_layer, merge_layer, n_layer_extractor, n_layer_aa, n_layer_gnn, n_layer_merger, smooth_term, smooth_method, 
                             Lambda, sep_sent, layer_norm, method_extractor, method_merger, lr, warmup_proportion, margin]]
    name = "_".join(name)
    return name
    
    
def sentence2ids(sentences, voc):

    def indexesFromSentence(voc, sentence):

        EOS_token = 1

        ids = []
        for word in sentence.split(' '):
            try:
                # 获得word在向量中的位置索引
                ids.append(voc.word2index[word])
            except:
                ids.append(2)
            
        ids.append(EOS_token)
        return ids

    ids = []
    for sentence in sentences:
        indexes_batch = [indexesFromSentence(voc, sentence)]
        ids.append(indexes_batch)
    
    return ids


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    # TODO 首先，我们有一个上下文（context）和四个选择（choice_1、choice_2、choice_3 和 choice_4）
    # TODO 对于给定的 ROC 示例，我们将创建以下四个输入：
    # Each choice will correspond to a sample on which we run the
    # inference. For a given roc example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # TODO 模型将为每个输入输出一个单一的分数
    # TODO 为了得到模型的最终决策，我们将对这四个输出运行 softmax 函数，以获得归一化的概率分布
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.

    features = []

    """
    examples数据组成：
        [
            {
                "context":[context_sentences, ...],
                "candidates":[ending, ...],
                "graph": ,
                "ith":,
                "ans":
            }
        ]
    """

    for example_index, example in enumerate(examples):
        context_sentences = example['context']

        """ ==================================== 处理上下文 ==================================== """
        context_tokens = []
        sentence_ind_context = []
        for ith_sent, sent in enumerate(context_sentences):
            # 中文是分成一个个中文字 ['这', '是', '一', '个', '测', '试', '.']
            sent_tokens = tokenizer.tokenize(sent)
            context_tokens = context_tokens + sent_tokens
            context_tokens = context_tokens + ['.']
            # ['0', '0', '0', '0', '0', '0', '0']
            sentence_ind_context.extend([ith_sent] * (len(sent_tokens) + 1))
        ith_sent = len(context_sentences) - 1
        """ ================================================================================== """
        # 收集特征
        choices_features = []
        # 设置是否放入最后要返回的特征组中
        if_append = True
        # 选择后续候选事件中最合理的事件
        for ending_index, ending in enumerate(example['candidates']):
            """ 上下文数据信息    创建副本便于修改 """
            context_tokens_tmp = copy.deepcopy(context_tokens)
            sentence_ind_context_tmp = copy.deepcopy(sentence_ind_context)
            """ ======================== 处理候选句子 ======================="""
            # 分割tokens
            ending_tokens = tokenizer.tokenize(ending)
            ending_tokens = ending_tokens + ['.']
            # tokens来自句子的编号
            sentence_ind_ending = [ith_sent + 1] * len(ending_tokens)
            """ =========================================================== """

            """ 在原地截断上下文和结束标记，使总长度小于或等于 max_seq_length - 3
                - 3 是因为考虑了 [CLS]、[SEP] 和 [SEP] 标记 
                修正tokens的长度，使其符合嵌入时输入的长度限制  """
            _truncate_seq_pair(context_tokens_tmp, ending_tokens, max_seq_length - 3)
            _truncate_seq_pair(sentence_ind_context_tmp, sentence_ind_ending, max_seq_length - 3)
            """ 根据示例中是否包含 ask_for，选择不同的标记顺序 """
            tokens = ["[CLS]"] + context_tokens_tmp + ["[SEP]"] + ending_tokens + ["[SEP]"]

            # segment_ids 是一个列表，用于标识上下文和候选（0 表示上下文，1 表示候选）
            segment_ids = [0] * (len(context_tokens_tmp) + 2) + [1] * (len(ending_tokens) + 1)
            # 更新 [CLS]上下文[SEP]候选句子[SEP] 的标记索引
            sentence_ind_context_tmp.insert(0, 0)
            sentence_ind_context_tmp.append(ith_sent)
            sentence_ind_ending.append(ith_sent + 1)
            # 拼接 ['0'([CLS]), '0', ..., '0', itn_sent] + [ith_sent + 1, ..., ith_sent + 1]
            sentence_ind = sentence_ind_context_tmp + sentence_ind_ending

            """ 将标记转换为 词汇表索引，并创建输入掩码 """
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # 对输入进行零填充，使其长度达到 max_seq_length
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            # 更新 sentence_ind，以考虑填充的影响
            sentence_ind += [p-1 for p in padding]

            graph = example['graph'][ending_index]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(sentence_ind) == max_seq_length

            # 将不同的上下文+候选句子的组合添加到 choices features 列表中
            # tokens=["[CLS]",'这', '是', '一', '个', '测', '试', '.', "[SEP]", '候', '选', '句', '子', '.', "[SEP]"]
            choices_features.append((tokens, input_ids, input_mask, segment_ids, sentence_ind, graph))

        answer = [0] * len(example['candidates'])
        # 将answer对应位置设置为1
        answer[example['ans']] = 1
        """ 将处理后的 InputFeatures 对象添加到 features中"""
        if if_append:
            features.append(
                InputFeatures(
                    example_id=example['ith'],
                    choices_features=choices_features,
                    answer=answer
                )
            )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """  主要是使得”上文“和”候选句子“的tokens总长度小于嵌入所能接受的最大长度-3  """
    while True:
        """ tokens_a是上下文    tokens_b是候选句子 """
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        # 比较A和B，则删除较长句子的最后一个token
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def ini_from_pretrained(config):
    pretrained_bert = torch.load("chinese-bert-wwm/pytorch_model.bin")
    # 将预训练的 BERT模型赋值给 state_dict
    state_dict = pretrained_bert
    """ 将不同的任务可能需要不同的模型配置放在一个配置里 """
    # 加载BERT模型的配置文件
    tmp_config = json.load(open('chinese-bert-wwm/config.json', 'r', encoding='utf-8'))
    # 一个用于存储和管理BERT模型配置的类 BertConfig
    bert_config = BertConfig(config)
    # 将本模型的配置中加入Bert的配置参数
    for key, value in tmp_config.items():
        bert_config.__dict__[key] = value

    graph_bert_config = bert_config
    """ 将config对象中的所有非私有属性添加到 graph_bert_config 中 """
    for k in dir(config):
        if "__" not in k:
            setattr(graph_bert_config, k, getattr(config, k))

    model_config = BertConfig(graph_bert_config)
    """ 这两行代码检查是否有链接预测的配置。如果有，就将其赋值给model_config """
    if config.link_predict:
        print("存在链接预测的配置")
        model_config = config.link_predict

    # 构造模型结构 graph_bert_model
    graph_bert_model = GraphBertModel(model_config)
    print("构造BertModel架构...")

    old_keys = []
    new_keys = []
    # state_dict 包含了从Bert模型文件中加载的参数
    for key in state_dict.keys():
        # 处理模型中的命名规则
        new_key = key
        if config.trained_model is None:
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = new_key.replace('beta', 'bias')
            if 'layer' in key:
                new_key = new_key.replace('layer', 'bert_layers')
            if 'bert.' in key:
                new_key = new_key.replace('bert.', '')
        else:
            if 'module.' in key:
                new_key = new_key.replace('module.', '')
        if new_key != key:
            old_keys.append(key)
            new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    for name, parameter in graph_bert_model.state_dict().items():
        if name in state_dict.keys():
            try:
                # bert_p 是预训练模型中与 name 对应的参数张量
                bert_p = state_dict[name]
                #  将 bert_p 的数据复制到 parameter 中
                parameter.data.copy_(bert_p.data)
            except:
                print('dimension mismatch! ' + name)
        else:
            pass
    # 状态字典（state_dict）是一个从参数名称映射到参数张量的字典
    graph_bert_model.keys_bert_parameter = state_dict.keys()
    return graph_bert_model


def freeze_params(model, requires_grad=False):
    freeze_ls = ['graph_extractor',
                   'adjacancy_approximator',
                   'gnn',
                   'merger_layers']
    for module in freeze_ls:
        
        parameters = getattr(model.encoder, module) 
        for param in parameters.parameters():
            param.requires_grad = requires_grad


def accuracy(out, labels):
    pdb.set_trace()
    out = np.array(out)
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
    
    
def loss_graph(appro_matrix, true_graph, loss_fn, smooth_term=0,  method='all'):
    if len(appro_matrix) == 2:
        
        graph_vector, graph_vector_post = appro_matrix
        assert graph_vector.shape == graph_vector_post.shape
        
        L = graph_vector.shape[1]
        
        loss_tot = 0
        
        for i in range(L):
            sigma = torch.ones_like(graph_vector[:,i,:])
            
            p = torch.distributions.Normal(graph_vector[:,i,:], sigma)
            q = torch.distributions.Normal(graph_vector_post[:,i,:], sigma)
            
            loss_tmp = loss_fn(p, q).sum()
            
            loss_tot = loss_tot + loss_tmp
            
    else:
        assert appro_matrix.shape == true_graph.shape
        
        L = appro_matrix.shape[1]
        loss_tot = 0
        if L < 10:
            for i in range(L):
                if method == 'all':
                    p = torch.distributions.categorical.Categorical(appro_matrix[:,i,:] + smooth_term)
                    q = torch.distributions.categorical.Categorical(true_graph[:,i,:] + smooth_term)
                else:
                    x = appro_matrix[:,i,:]
                    y = true_graph[:,i,:]
                    x[:, i] += 1
                    y[:, i] += 1
                    
                    p = torch.distributions.categorical.Categorical(x)
                    q = torch.distributions.categorical.Categorical(y)
                    
                loss_tmp = loss_fn(p, q).sum()
                '''
                if smooth_term  == 0:
                    loss_tmp = loss_fn(q, p).sum()
                else:
                    loss_tmp = loss_fn(p, q).sum()
                '''
                #loss_tmp = -(appro_matrix[:,i,:].log() * true_graph[:,i,:]).sum()
                loss_tot = loss_tot + loss_tmp
        else:
            loss_fn = torch.nn.MSELoss()
            
            loss_tot = loss_fn(appro_matrix, true_graph)
            
    return loss_tot
    
    
def write_result_to_file(args,result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(result+"\n")


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def do_evaluation(model, eval_dataloader, opt, gpu_ls=None,  output_res=False):
    model.eval()
    # eval_dataloader：用于评估的数据加载器，包含评估数据集。
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # logits_all = None
    if gpu_ls:
        gpu_id = gpu_ls[0]
    else:
        gpu_id = opt.gpuid
    # 临时禁用梯度计算
    with torch.no_grad():
        # res = {'ids':[],'pred':[],'ans':[]}
        res = pd.DataFrame()
        for dat in eval_dataloader:
            dat = [t.cuda(gpu_id) for t in dat]

            example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers = dat

            answers = answers
            num_choices = input_ids.shape[1]
            cls_score = []
            res_tmp = []
            for n in range(num_choices):
                input_ids_tmp = input_ids[:, n, :]
                segment_ids_tmp = segment_ids[:, n, :]
                sentence_inds_tmp = sentence_inds[:, n, :]
                graphs_tmp = graphs[:, n, :]
                graphs_tmp_scaled = graphs_tmp

                _, cls_score_tmp, attn_scores = model(input_ids=input_ids_tmp,
                                                      token_type_ids=segment_ids_tmp,
                                                      sentence_inds=sentence_inds_tmp,
                                                      graphs=graphs_tmp_scaled)
                cls_score_tmp = cls_score_tmp.softmax(-1)
                cls_score.append(cls_score_tmp.detach().cpu().numpy()[:, 1].tolist())
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:, 0].tolist())
                res_tmp.append(cls_score_tmp.detach().cpu().numpy()[:, 1].tolist())

            cls_score = np.array(cls_score).T
            answers = answers.detach().cpu().numpy()
            num_acc_tmp = sum(cls_score.argmax(1) == answers.argmax(1))

            res_tmp.append(answers.argmax(1))
            res_tmp = pd.DataFrame(np.array(res_tmp).T)
            res = res._append(res_tmp)
            eval_accuracy += num_acc_tmp

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    eval_accuracy = float(eval_accuracy) / nb_eval_examples

    model.zero_grad()
    if not output_res:
        return eval_accuracy, cls_score, answers
    else:
        return eval_accuracy, res, cls_score, answers


def graph_ids_to_tensor(all_graph_ids, opt):
    if 'mcnc' in opt.train_data_dir:
        max_L = 7
    elif 'roc' in opt.train_data_dir:
        max_L = 15
        
    PAD_token = 2
    all_graph_ids_padded = []
    for sample in all_graph_ids:
        all_graph_ids_padded.append([])
        for candidate in sample:
            all_graph_ids_padded[-1].append([])
            for sentence in candidate:
                l_sent = len(sentence[0])
                if l_sent > max_L:
                    sentence[0] = sentence[0][:max_L]
                elif l_sent < max_L:
                    l_diff = max_L - l_sent
                    pad_ls = [PAD_token] * l_diff
                    
                    sentence[0] = sentence[0] + pad_ls
                all_graph_ids_padded[-1][-1].append(sentence)
             
    all_graph_ids_padded = torch.LongTensor(all_graph_ids_padded)
    all_graph_ids_padded = all_graph_ids_padded.squeeze()
    
    return all_graph_ids_padded
    
    
def retro(key, graph):
    key_expand = [key + "_obj", key + '_subj']
    res = []
    for key_tmp in key_expand:
        try:
            fwd_nodes_tmp = graph[key_tmp]
            res.append(fwd_nodes_tmp)
        except:
            pass
            
    return res


def load_data(features, batch_size=32, dat_type='train'):
    example_ids = torch.tensor([feature.example_id for feature in features], dtype=torch.long)
    input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    input_masks = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    sentence_inds = torch.tensor(select_field(features, 'sentence_ind'), dtype=torch.long)
    graphs = select_field(features, 'graph')  # #
    if graphs[0][0] is not None:
        graphs = np.array(graphs)
        graphs = torch.tensor(graphs, dtype=torch.float)  # #

    answers = torch.tensor([f.answer for f in features], dtype=torch.long)
    data = TensorDataset(example_ids, input_ids, input_masks, segment_ids, sentence_inds, graphs, answers)

    if dat_type == 'train':
        data_sampler = RandomSampler(data)
    else:
        data_sampler = SequentialSampler(data)
    # PyTorch 中的数据加载器，用于将数据集封装成可迭代对象，以便在模型训练和评估过程中按批次加载数据
    dataloader = DataLoader(data, sampler=data_sampler, batch_size=batch_size)
    return dataloader


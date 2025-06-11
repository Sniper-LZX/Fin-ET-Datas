import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BertModules import *
import pdb        

# nn.Module 是PyTorch中提供的一个基本的神经网络模块类

"""
这个模块包含了一个全连接层、一个可选的dropout层、一个激活函数，以及可选的层归一化，
这些层按顺序应用于输入的隐藏状态，最后输出处理后的隐藏状态
"""
class GATSelfOutput(nn.Module):
    def __init__(self, config):
        super(GATSelfOutput, self).__init__()
        # 这里创建了一个线性变换层（也称为全连接层或者密集层），其输入和输出特征维度都设置为 config.hidden_size。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 是否需要归一化
        self.layer_norm = config.layer_norm
        # ACT2FN 是一个将函数名映射到函数的字典
        self.act_fn = ACT2FN[config.act_fn_branch]
        if self.layer_norm:
            # 创建一个归一化层 BertLayerNorm
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        try:
            # 正则化，在训练过程中随机地使网络层中的一部分神经元失活
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except:
            # 若上面出错（可能是没有这个属性值），则默认的dropout为0.1
            self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        if self.layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


"""
    通过结合 BERT 的自注意力和跨注意力机制，计算和更新图中的节点表示，以实现节点特征的聚合和处理。
    包含GATSelfOutput
    参数
        graph_vectors, context_vectors, attention_scores, drop_first_token, sent_ind, attention_mask
    来更新
        graph_vectors
"""
class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        # 读取方法
        self.method = config.method
        # TODO 根据method的值，GATLayer可以创建不同类型的注意力层
        if self.method == 'self':
            # 使用 BertSelfAttention 创建自注意力层
            self.attn_layer = BertSelfAttention(config)
        elif self.method == 'cross':
            # 使用 BertCrossAttention 创建跨注意力层
            self.attn_layer = BertCrossAttention(config)
        else:
            # 使用pdb.set_trace()进入Python的调试器，这通常表示配置有误
            pdb.set_trace()
        # TODO 它创建一个 GATSelfOutput 实例，用于处理经过注意力层后的节点表示，输出处理后的隐藏状态
        self.output = GATSelfOutput(config)

    def forward(self, graph_vectors, context_vectors=None, attention_scores=None, drop_first_token=True, sent_ind=None, attention_mask=None):
        # 如果是自注意力层
        # TODO 它会使用graph_vectors和可选的attention_scores来计算新的节点表示
        if isinstance(self.attn_layer, BertSelfAttention):
            graph_vectors = self.attn_layer(graph_vectors, attention_probs=attention_scores)
        # 如果是交叉注意力层
        # TODO 则会结合context_vectors、attention_mask等参数来计算
        elif isinstance(self.attn_layer, BertCrossAttention):
            graph_vectors = self.attn_layer(graph_vectors, context_vectors, drop_first_token=drop_first_token, sent_ind=sent_ind, attention_mask=attention_mask)

        graph_vectors = self.output(graph_vectors)

        return graph_vectors


"""
    从输入的上下文向量中提取图向量（句子向量 提取出 图信息） 返回图向量
    包含了GATLayer
"""
class GraphExtractor(nn.Module):
    def __init__(self, config):
        super(GraphExtractor, self).__init__()
        # 图提取层的数量
        self.num_layers = config.n_layer_extractor
        # 对其的修改不会影响原始的 config 对象
        self.config = copy.deepcopy(config)
        # 使用哪种图提取方法
        self.config.method = config.method_extractor
        # 是否在层次间使用了层归一化技术
        self.config.layer_norm = config.layer_norm
        # 用来指示 GraphExtractor 是否被用于链接预测任务
        self.is_link_prediction = config.link_predict
        # 这行创建了一个 GATLayer 实例，这是一种基于注意力机制的图网络层
        layer = GATLayer(self.config)
        # 通过列表推导式创建了 self.config.n_layer_extractor 指定数量的层
        self.extract_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.config.n_layer_extractor)])
        
    def forward(self, sent_ind, start_layer, subsequent_layers=None):
        '''
        start_layer: tensor with shape batch * seq_length * dim
                       seq_length = number of tokens
        subsequent_layers: list, each element is a tensor with shape batch * seq_length * dim
        graph_vectors: tensor with shape batch * seq_length * dim
                       seq_length = number of sentences


            start_layer: 一个形状为 batch * seq_length * dim 的张量，代表每个句子初始层的向量。seq_length = tokens的数量
            subsequent_layers: 一个列表，其中每个元素都是一个形状为 batch * seq_length * dim 的张量，代表随后每个层的向量。
        '''
        assert len(subsequent_layers) == self.num_layers

        batch = sent_ind.shape[0]
        # TODO 如果不进行链接预测
        if not self.is_link_prediction:
            num_sent = sent_ind.max().detach().cpu().numpy() + 1
            
            graph_vectors = []
            # 遍历每个批次的数据，从 start_layer 张量中取出对应的句子向量，并且对于每个句子中的向量，通过取均值得到一个压缩的句子表示
            for n in range(batch):
                graph_vectors_sample = []
                for ith_sent in range(num_sent):
                    if ith_sent != 0:
                        # start_layer[n]获得的是对于一个句子的关于tokens、位置、归属句子索引的嵌入
                        graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent].mean(0).unsqueeze(0)
                    else:
                        '''
                        The first [CLS] token should not be taken into consideration of the first sentence.
                        '''
                        # 如果是第一句话的话，会排除掉第一个[CLS]标记。
                        graph_vectors_sample_ith = start_layer[n][sent_ind[n] == ith_sent][1:].mean(0).unsqueeze(0)
                    # 然后将所有句子的表示向量组合起来形成一个图向量
                    graph_vectors_sample.append(graph_vectors_sample_ith)
                try:
                    # 使用 torch.cat(graph_vectors_sample).unsqueeze(0) 将一个批次中所有句子的向量连接起来，形成一个张量，
                    # 并在最前面增加一维，代表批次
                    graph_vectors_sample = torch.cat(graph_vectors_sample).unsqueeze(0)
                except:
                    # 如果发生异常，使用 pdb.set_trace() 进入Python调试器以便进一步调试
                    pdb.set_trace()
                # 将每个批次的图向量添加到 graph_vectors 列表
                graph_vectors.append(graph_vectors_sample)
            # 使用 torch.cat(graph_vectors) 将所有批次的图向量连接起来
            graph_vectors = torch.cat(graph_vectors)
            for i in range(self.num_layers):
                graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i],  sent_ind=sent_ind)                
        # TODO 如果进行链接预测
        else:
            #pdb.set_trace()
            # 则简单地将 start_layer 赋值给 graph_vectors
            graph_vectors = start_layer
            # 通过每一个 extract_layers[i] 层进行一次传播
            for i in range(self.num_layers):
                graph_vectors = self.extract_layers[i](graph_vectors, subsequent_layers[i])
        return graph_vectors
    

class AdjacancyApproximator(nn.Module):
    """
        通过多层 GAT 层和一个最终的自注意力层，对输入的图向量进行处理，
        近似计算一个邻接矩阵，这通常用在图网络中表示节点之间的连接关系

        包含了GATLayer、BertSelfAttention

        输出：attn_scores（注意力分数）, graph_vectors（更新后的图向量）
        注意力分数 attn_scores 代表着图中节点之间的关系强度，可以被视为是邻接矩阵的一种近似表示
    """
    def __init__(self, config):
        super(AdjacancyApproximator, self).__init__()
        
        self.num_layers = config.n_layer_aa - 1
        
        self.config = copy.deepcopy(config)

        # TODO 创建自注意力层
        self.config.method = 'self'
        self.config.layer_norm = config.layer_norm
        layer = GATLayer(self.config)
        # 关闭其注意力机制的dropout，不使用 dropout 正则化
        layer.attn_drop = False

        # 构建 aa_layers（邻接矩阵近似层）
        self.aa_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])
        # self.config_final_layer = copy.deepcopy(config)

        # 创建一个 BertSelfAttention 实例作为最终的层
        self.final_layer = BertSelfAttention(config)
        # 确保输出注意力分数
        self.final_layer.output_attentions = True
        # 启用特定模式
        self.final_layer.unmix = True
        # 关闭注意力机制的 dropout
        self.final_layer.attn_drop = False
        
    def forward(self, graph_vectors):
        # 该方法接收一个参数 graph_vectors，表示输入的图向量
        # 历 aa_layers中的每一个GATLayer
        for i in range(self.num_layers):
            # 把 graph_vectors 作为输入逐层传递下去。
            # 每一层的输出都将作为下一层的输入。
            graph_vectors = self.aa_layers[i](graph_vectors)
        attn_scores, graph_vectors = self.final_layer(graph_vectors)  
        # attn_scores 包含了最终层中的注意力分数，而 graph_vectors 则是经过所有处理层后的图向量。
        return attn_scores, graph_vectors


class GNN(nn.Module):
    """
        GNN的实现
        使用多个 GAT 层来逐步更新和聚合图向量
        包含GATLayer
    """
    def __init__(self, config):
        super(GNN, self).__init__()
        
        self.config = copy.deepcopy(config)

        # 读取配置中的图神经网络层数
        self.num_layers = config.n_layer_gnn
        # 将配置中的layer_norm值也保存在类的配置中
        self.config.layer_norm = config.layer_norm
        # 配置中指定的 method_gnn（图神经网络方法)
        if config.method_gnn == 'gat':
            self.config.method = 'self'
            # 创建一个GATLayer的实例
            layer = GATLayer(self.config)
            # 不对注意力权重应用dropout
            layer.attn_drop = False
            self.gnn_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])
        elif config.method_gnn == 'ggnn':
            raise NotImplementedError
            
    def forward(self, attn_scores, graph_vectors):
        
        if self.config.method_gnn == 'gat':
            # 通过第一个GAT层传递原始图向量和注意力分数attn_scores
            graph_vectors = self.gnn_layers[0](graph_vectors, attention_scores=attn_scores)
            # 接下来的每一层仅处理前一层的输出，每一层的输入都是上一层的输出图向量
            for i in range(1, self.num_layers, 1):
                graph_vectors = self.gnn_layers[i](graph_vectors)
        
        elif self.config.method == 'ggnn':
            raise NotImplementedError
        return graph_vectors        
    

class GATResMergerLayer(nn.Module):
    """
        在 BERT 模型中引入图注意力网络（GAT）层。
        聚合向量表示作为事件图的表示（猜测）
        包含了GATLayer、BertLayerNorm
    """
    def __init__(self, config):
        super(GATResMergerLayer, self).__init__()
        self.config = copy.deepcopy(config)
        # 是否使用层归一化
        self.config.layer_norm = config.layer_norm
        # 目的是为了创建跨注意力层
        config.method = 'cross'
        config.layer_norm = False  # !!
        # 使用修改后的 config 创建一个 GATLayer 实例，且不使用归一化
        layer = GATLayer(config)
        self.layer = layer
        # 表示这一层将输出注意力权重
        self.layer.output_attentions = True
        # 如果 self.config.layer_norm 为真，则创建一个 BertLayerNorm 实例用于层归一化。
        # BertLayerNorm 是一种特定的归一化层，这里基于配置中的hidden_size和layer_norm_eps参数来初始化
        if self.config.layer_norm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, context_vectors, graph_vectors, sent_ind, attention_mask=None):
        # 上下文向量、图向量、句子索引和注意力掩码
        # TODO 通过GAT层处理 context_vectors和 graph_vectors   输出更新后的上下文向量
        # 参数 drop_first_token=False 指示不丢弃第一个令牌
        context_vectors_updated = self.layer(context_vectors, graph_vectors, sent_ind,
                                             attention_mask=attention_mask, drop_first_token=False)
        # 如果启用了归一化
        if self.config.layer_norm:
            # 再次通过BertLayerNorm进行归一化处理
            context_vectors_updated = BertLayerNorm(context_vectors + context_vectors_updated)
        else:
            # 直接向量相加
            context_vectors_updated = context_vectors + context_vectors_updated
        return context_vectors_updated


class AddMergerLayer(nn.Module):
    """
        用于合并来自不同信息源（例如上下文和图结构）的表示
    """
    def __init__(self, config):
        super(AddMergerLayer, self).__init__()
        self.config = copy.deepcopy(config)

    def forward(self, context_vectors, graph_vectors, sent_ind):
        graph_vectors_spanned = graph_vectors, sent_ind
        context_vectors_updated = context_vectors + graph_vectors_spanned
        return context_vectors_updated
        

class GraphBertEncoder(nn.Module):
    """
        这段代码结合了BERT层和图神经网络（GNN）层，首先通过BERT层处理输入，
        然后提取图结构向量，并通过图神经网络进行处理，
        最后继续通过剩余的BERT层处理。
        整个过程中的隐藏状态和注意力分数被存储，并根据需要返回。
    """
    def __init__(self, config, output_attentions, keep_multihead_output):
        super(GraphBertEncoder, self).__init__()
        self.config = config
        self.output_attentions = output_attentions
        # 通过 BertLayer类 创建的一个BERT层实例
        bert_layer = BertLayer(config, output_attentions=output_attentions,
                                  keep_multihead_output=keep_multihead_output)
        # 创建了一个包含多个 BertLayer层的列表。
        self.bert_layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(config.num_hidden_layers)])
        # 用于从输入中提取图相关的特征
        self.graph_extractor = GraphExtractor(config)
        # 创建一个似乎用于估计或近似邻接关系的 AdjacancyApproximator 实例
        self.adjacancy_approximator = AdjacancyApproximator(config)
        # 创建一个图神经网络（GNN）的实例，它可以处理图结构数据
        self.gnn = GNN(config)
        # 创建一个 GATSelfOutput 实例；根据命名，这可能是一种自注意力机制的输出转换层
        self.post_converter = GATSelfOutput(config)  # !!!
        # 创建指定数量次，并存储到 self.merger_layers 模块列表中
        if config.method_merger == 'gat':
            # 创建一个图注意力（GATResMergerLayer）融合层
            merger_layer = GATResMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])
        elif config.method_merger == 'add':
            # 创建一个加和（AddMergerLayer）融合层，上下文表示和图表示相加
            merger_layer = AddMergerLayer(config)
            self.merger_layers = nn.ModuleList([merger_layer for _ in range(config.n_layer_merger)])

    def forward(self, hidden_states, attention_mask, sentence_ind, true_adjacancy_matrix=None, output_all_encoded_layers=True, head_mask=None):
        # 存储所有的编码器层
        all_encoder_layers = []
        # 所有注意力
        all_attentions = []

        """
            hidden_states: 输入的隐藏状态（通常是词嵌入）。
            attention_mask: 注意力掩码，用于屏蔽填充位置。
            sentence_ind: 句子索引，用于图结构提取。
            true_adjacancy_matrix: 真实的邻接矩阵（可选），用于图神经网络。
            output_all_encoded_layers: 是否输出所有编码层。
            head_mask: 注意力头的掩码。
        """

        start_layer = self.config.start_layer - 1
        merge_layer = self.config.merge_layer - 1
        
        num_tot_layers = len(self.bert_layers)
        num_sub_layers = self.config.n_layer_extractor
        num_merger_layers = self.config.n_layer_merger
        
        assert start_layer + num_sub_layers <= merge_layer

        """
            start_layer: 开始层的索引。
            merge_layer: 合并层的索引。
            num_tot_layers: BERT层的总数。
            num_sub_layers: 提取子层的数量。
            num_merger_layers: 合并层的数量。
            断言确保开始层和子层数之和不超过合并层。
        """
        
        def append(hidden_states):
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            all_encoder_layers.append(hidden_states)

        # 遍历合并层之前的所有层，更新 hidden_states 并调用 append 存储结果。
        # 输入的隐藏状态（通常是词嵌入）经过多层 BertLayer 的处理，最终得到多层的隐藏状态表示
        for i in range(merge_layer):
            # 获得对每个 token的嵌入
            hidden_states = self.bert_layers[i](hidden_states, attention_mask, head_mask[i])
            append(hidden_states)

        # 提取开始层和随后的子层的上下文向量
        context_vector_start = all_encoder_layers[start_layer]
        context_vector_subsequent = all_encoder_layers[(start_layer + 1): (start_layer + 1 + num_sub_layers)]

        # 使用 graph_extractor 提取图结构向量
        # 使用 adjacancy_approximator 计算注意力分数
        graph_vectors = self.graph_extractor(sentence_ind, context_vector_start, context_vector_subsequent)
        attn_scores, _ = self.adjacancy_approximator(graph_vectors)

        # 如果提供了真实的邻接矩阵，使用矩阵与图向量相乘并进行后处理。
        # 否则，直接使用邻接矩阵通过 gnn 更新图向量
        if true_adjacancy_matrix is not None:
            true_adjacancy_matrix = true_adjacancy_matrix.unsqueeze(1)
            true_adjacancy_matrix = true_adjacancy_matrix.squeeze()
            
            graph_vectors = self.gnn(attn_scores, graph_vectors)
            true_adjacancy_matrix = true_adjacancy_matrix.squeeze() 

        else:
            graph_vectors = self.gnn(attn_scores, graph_vectors)

        #  遍历合并层，更新 hidden_states
        #  如果合并方法不是 'combine'，分别处理 hidden_states 和 graph_vectors。
        #  如果是 'combine'，拼接 attention_mask 和 hidden_states。
        for ith_merger_layer, jth_bert_layer in zip(list(range(num_merger_layers)), list(range(merge_layer, merge_layer + num_merger_layers))):
            if self.config.method_merger != 'combine':
                hidden_states = self.merger_layers[ith_merger_layer](hidden_states, graph_vectors, sentence_ind)
                hidden_states = self.bert_layers[jth_bert_layer](hidden_states, attention_mask, head_mask[jth_bert_layer])
            else:
                attention_mask_tmp = torch.cat([attention_mask, attention_mask[:, :, :, :(graph_vectors.shape[1] + 0)] * 0], -1) 
                hidden_states = torch.cat([hidden_states, graph_vectors], 1)
                hidden_states = self.bert_layers[jth_bert_layer](hidden_states, attention_mask_tmp, head_mask[jth_bert_layer])
                hidden_states = hidden_states[:, :-(graph_vectors.shape[1] + 0), :]
                
            append(hidden_states)

        # 遍历剩余的层，更新 hidden_states 并调用 append 存储结果。
        for j in range(merge_layer + num_merger_layers, num_tot_layers):
            hidden_states = self.bert_layers[j](hidden_states, attention_mask, head_mask[j])
            append(hidden_states)

        # 根据 output_attentions 和 true_adjacancy_matrix 的值，返回相应的结果。
        if self.output_attentions:
            if true_adjacancy_matrix is not None:
                return all_attentions, all_encoder_layers, attn_scores
            else:
                return all_attentions, all_encoder_layers, attn_scores
        else:
            if true_adjacancy_matrix is not None:
                return all_encoder_layers, attn_scores
            else:
                return all_encoder_layers, attn_scores


class GraphBertModel(BertPreTrainedModel):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(GraphBertModel, self).__init__(config)

        self.config = config
        # 如果 True, 同时输出由模型计算的每一层的 attentions weights
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        # encoder编码器
        self.encoder = GraphBertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        # 初始化一个池化层
        self.pooler = BertPooler(config)
        # 初始化用于下一句预测（NSP）任务的层。在BERT的原始预训练过程中，NSP任务是判断两个句子是否是连续的文本。
        self.cls = BertOnlyNSPHead(config)
        # 应用权重初始化
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]
    ##

    def forward(self, input_ids, sentence_inds=None, graphs=None, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, head_mask=None):
        if attention_mask is None:
            # print("构造attention_mask")
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            # print("构造token_type_ids")
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        # TODO 根据 input_ids获得单词的嵌入 （对应 BertModules的 BertEmbeddings）
        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoded_layers = self.encoder(hidden_states=embedding_output,
                                      attention_mask=extended_attention_mask,
                                      sentence_ind=sentence_inds,
                                      true_adjacancy_matrix=graphs,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers, attn_scores = encoded_layers
        else:
            encoded_layers, attn_scores = encoded_layers
        # 编码器的最后一层输出()
        sequence_output = encoded_layers[-1]

        # 对 sequence_output 进行池化操作
        pooled_output = self.pooler(sequence_output)
        # 使用下一句预测头部对 pooled_output 进行预测
        cls_scores = self.cls(pooled_output)
        # 是否返回所有编码层的输出
        if not output_all_encoded_layers:
            # 只返回最后一层的编码输出
            encoded_layers = encoded_layers[-1]
        # 是否返回所有注意力得分
        if self.output_attentions:
            return all_attentions, encoded_layers, cls_scores, attn_scores
        else:
            return encoded_layers, cls_scores, attn_scores

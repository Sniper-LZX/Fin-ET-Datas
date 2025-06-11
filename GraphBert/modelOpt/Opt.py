

def model_opts(parser):
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="训练数据目录的路径")

    parser.add_argument("--test_data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="测试数据目录的路径")

    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        required=False,
                        help="预训练的 BERT模型的路径")

    parser.add_argument("--bert_tokenizer",
                        default=None,
                        type=str,
                        required=False,
                        help="BERT分词器的路径")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="存储模型检查点的目录")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="输入序列的最大总长度")

    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--pret",
                        default=True,
                        action='store_true',
                        help="是否对数据进行预处理")

    parser.add_argument("--pretrain",
                        default=False,
                        action='store_true',
                        help="是否预训练")

    parser.add_argument("--link_predict",
                        default=False,
                        action='store_true',
                        help="是否链接预测")

    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="是否评估开发级")

    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--do_margin_loss",
                        default=False,
                        action='store_true',
                        help="Use margin loss or log-loss.")

    parser.add_argument('--margin',
                        type=float, default=0.15,
                        help='Margin value used in the MultiMarginLoss.')

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=6.25e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # 是否在 CPU 上执行优化并保留优化器的平均值
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    # 是否使用 16 位浮点精度而不是 32 位浮点精度
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--l2_reg',
                        type=float, default=0.01,
                        help='Margin value used in the MultiMarginLoss.')
    parser.add_argument('--gpuls', type=int, default=0, help='The gpus to use')
    parser.add_argument('--gpuid', type=int, default=0, help='The gpu to use')
    
    parser.add_argument("--start_layer",
                        default=6,
                        type=int,
                        help="The layer starting from which the graph vectors extracts information from BERT.")
                             
    parser.add_argument("--merge_layer",
                        default=10,
                        type=int,
                        help="The layer which auxilary branch merge with the stem.")
                        
    parser.add_argument("--n_layer_extractor",
                        default=2,
                        type=int,
                        help="Number of layers for extracting graph information from BERT.")
                             
    parser.add_argument("--n_layer_aa",
                        default=2,
                        type=int,
                        help="Number of layers for approximating adjacancy matrix using extracted graph vectors.")
                             
    parser.add_argument("--n_layer_gnn",
                        default=2,
                        type=int,
                        help="Number of layers for GNN to cunduct reasoning using extracted graph vectors and (approximated) adjacancy matrix.")

    parser.add_argument("--n_layer_merger",
                        default=2,
                        type=int,
                        help="Number of merger layers for merging extracted graph vectors and hidden states of BERT.")

    parser.add_argument("--method_extractor",
                        default='cross',
                        type=str,
                        help="Type of extracting graph vector from BERT.")
                        
    parser.add_argument("--method_gnn",
                        default='gat',
                        type=str,
                        help="Type of GNN for conducting graph reasoning using extracted graph vectors and (approximated) adjacancy matrix.")
                        
    parser.add_argument("--method_merger",
                        default='gat',
                        type=str,
                        help="Way for merging GNN extracted features into BERT decoder.")
                        
    parser.add_argument("--layer_norm",
                        default=False,
                        action='store_true',
                        help="If conducting layer normalization for the auxiliary graph branch.")

    parser.add_argument("--act_fn_branch",
                        default='gelu',
                        type=str,
                        help="If conducting layer normalization for the auxiliary graph branch.")
    parser.add_argument("--num_frozen_epochs",
                        default=1,
                        type=int,
                        help="The number of epochs in which the parameters of bert model is frozen.")
    parser.add_argument("--use_bert",
                        default=False,
                        action='store_true',
                        help="If use pretrained bert model.使用预训练的bert模型")
    parser.add_argument("--loss_aa_smooth",
                        default=0,
                        type=float,
                        help="A constant for smoothing the aa_loss_function")
    parser.add_argument("--loss_aa_smooth_method",
                        default='all',
                        type=str,
                        help="A constant for smoothing the aa_loss_function")
    parser.add_argument("--Lambda",
                        default=0.0001,
                        type=float,
                        help="Lambda for graph loss")
                        
    parser.add_argument("--sep_sent",
                        default=False,
                        action='store_true',
                        help="Wether to seperate sentences in graph extractor")
                        
    parser.add_argument("--trained_model", default=None, type=str, required=False,
                        help="The path of trained model.")
                        
    parser.add_argument("--graph_embedder_voc_path", default=None, type=str, required=False,
                        help="The path of vocabulary graph embedder.")

    parser.add_argument("--graph_embedder_path", default=None, type=str, required=False,
                        help="The path of graph embedder.")
                        
    parser.add_argument("--baseline", default=False, action='store_true',
                        help="Training baseline model or GraphBert")

    parser.add_argument("--path_gb", default=None, type=str, required=False,
                        help="Path of pertrained GraphBert")
    parser.add_argument("--path_bt", default=None, type=str, required=False,
                        help="Path of pertrained BERT")
    parser.add_argument("--path_er", default=None, type=str, required=False,
                        help="Path of pertrained ERNIE")
    parser.add_argument("--path_gt", default=None, type=str, required=False,
                        help="Path of pertrained GraphTransformer")

    parser.add_argument('--nell_embed_dir',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/kb_embedding/nell_concept2vec.txt',
                        type=str, required=False)
    parser.add_argument('--content_dim', default=100, type=int, required=False)
    parser.add_argument('--num_content', default=27, type=int, required=False)
    parser.add_argument('--doc_len', default=317, type=int, required=False)
    parser.add_argument('--seq_len', default=384, type=int, required=False)
    parser.add_argument('--question_len', default=64, type=int, required=False)
    parser.add_argument('--wordnet_content_dir',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/retrieval/wordnet/retrived_synsets.data',
                        type=str, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)

    parser.add_argument('--wordnet', default=False, type=bool, required=False)
    parser.add_argument('--nell', default=False, type=bool, required=False)
    parser.add_argument('--wordnet_embed_dir',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/kb_embedding/wn_concept2vec.txt',
                        type=str, required=False)
    parser.add_argument('--nell_content_dir',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/retrieval/nell/',
                        type=str, required=False)

    parser.add_argument('--dev',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/tokens/dev.tokenization.uncased.data',
                        type=str, required=False)
    parser.add_argument('--train',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/tokens/train.tokenization.cased.data',
                        type=str, required=False)
    parser.add_argument('--dev_answer',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/data/dev.json',
                        type=str, required=False)
    parser.add_argument('--train_answer',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/data/train.json',
                        type=str, required=False)
    parser.add_argument('--num_layers',
                        default=12,
                        type=int, required=False)
    parser.add_argument('--num_gpus',
                        default=8,
                        type=int, required=False)

    parser.add_argument('--train_subanswer',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/sub_answer/train_subanswers.json',
                        type=str, required=False)

    parser.add_argument('--dev_subanswer',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/sub_answer/dev_subanswers.json',
                        type=str, required=False)

    parser.add_argument('--dev_gold_file',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/orig_data/dev.json',
                        type=str, required=False)
    parser.add_argument('--dev_prediction_file',
                        default='/users5/kxiong/work/GraphBert/knowledge_graph/eval_output/dev.json',
                        type=str, required=False)
    parser.add_argument('--freeze',
                        default=False,
                        type=bool, required=False)
    parser.add_argument('--model_name',
                        default='gpt2',
                        type=str, required=False)

     





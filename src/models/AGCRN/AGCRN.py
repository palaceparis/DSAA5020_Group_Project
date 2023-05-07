import torch
import torch.nn as nn
from AGCRNCell import AGCRNCell


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
            )

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))  # type: ignore
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_node, args.embed_dim), requires_grad=True
        )

        self.encoder = AVWDCRNN(
            args.num_nodes,
            args.input_dim,
            args.rnn_units,
            args.cheb_k,
            args.embed_dim,
            args.num_layers,
        )

        # predictor
        self.end_conv = nn.Conv2d(
            1,
            args.horizon * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(
            source, init_state, self.node_embeddings
        )  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(
            -1, self.horizon, self.output_dim, self.num_node
        )
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output


# # Visualize the model
# import torch
# from torchviz import make_dot
# import argparse
# import configparser
# from torch.utils.tensorboard.writer import SummaryWriter


# # Parse command-line arguments
# # parser

# Mode = "Train"
# DEBUG = "False"
# DATASET = "CARBON"
# DEVICE = "cuda:0"
# MODEL = "AGCRN"

# # get configuration
# # config_file = './{}_{}.conf'.format(DATASET, MODEL)
# config_file = "src/models/AGCRN/CARBON.conf"
# # print('Read configuration file: %s' % (config_file))
# config = configparser.ConfigParser()
# config.read(config_file)
# args = argparse.ArgumentParser(description="arguments")
# args.add_argument("--dataset", default=DATASET, type=str)
# args.add_argument("--mode", default=Mode, type=str)
# args.add_argument("--device", default=DEVICE, type=str, help="indices of GPUs")
# args.add_argument("--debug", default=DEBUG, type=eval)
# args.add_argument("--model", default=MODEL, type=str)
# args.add_argument("--cuda", default=False, type=bool)
# # data
# args.add_argument("--val_ratio", default=config["data"]["val_ratio"], type=float)
# args.add_argument("--test_ratio", default=config["data"]["test_ratio"], type=float)
# args.add_argument("--lag", default=config["data"]["lag"], type=int)
# args.add_argument("--horizon", default=config["data"]["horizon"], type=int)
# args.add_argument("--num_nodes", default=config["data"]["num_nodes"], type=int)
# args.add_argument("--tod", default=config["data"]["tod"], type=eval)
# args.add_argument("--normalizer", default=config["data"]["normalizer"], type=str)
# args.add_argument("--column_wise", default=config["data"]["column_wise"], type=eval)
# args.add_argument("--default_graph", default=config["data"]["default_graph"], type=eval)
# # model
# args.add_argument("--input_dim", default=config["model"]["input_dim"], type=int)
# args.add_argument("--output_dim", default=config["model"]["output_dim"], type=int)
# args.add_argument("--embed_dim", default=config["model"]["embed_dim"], type=int)
# args.add_argument("--rnn_units", default=config["model"]["rnn_units"], type=int)
# args.add_argument("--num_layers", default=config["model"]["num_layers"], type=int)
# args.add_argument("--cheb_k", default=config["model"]["cheb_order"], type=int)
# # train
# args.add_argument("--loss_func", default=config["train"]["loss_func"], type=str)
# args.add_argument("--seed", default=config["train"]["seed"], type=int)
# args.add_argument("--batch_size", default=config["train"]["batch_size"], type=int)
# args.add_argument("--epochs", default=config["train"]["epochs"], type=int)
# args.add_argument("--lr_init", default=config["train"]["lr_init"], type=float)
# args.add_argument("--lr_decay", default=config["train"]["lr_decay"], type=eval)
# args.add_argument(
#     "--lr_decay_rate", default=config["train"]["lr_decay_rate"], type=float
# )
# args.add_argument("--lr_decay_step", default=config["train"]["lr_decay_step"], type=str)
# args.add_argument("--early_stop", default=config["train"]["early_stop"], type=eval)
# args.add_argument(
#     "--early_stop_patience", default=config["train"]["early_stop_patience"], type=int
# )
# args.add_argument("--grad_norm", default=config["train"]["grad_norm"], type=eval)
# args.add_argument("--max_grad_norm", default=config["train"]["max_grad_norm"], type=int)
# args.add_argument("--teacher_forcing", default=False, type=bool)
# # args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
# args.add_argument(
#     "--real_value",
#     default=config["train"]["real_value"],
#     type=eval,
#     help="use real value for loss calculation",
# )
# # test
# args.add_argument("--mae_thresh", default=config["test"]["mae_thresh"], type=eval)
# args.add_argument("--mape_thresh", default=config["test"]["mape_thresh"], type=float)
# # log
# args.add_argument("--log_dir", default="outputs/AGCRN", type=str)
# args.add_argument("--log_step", default=config["log"]["log_step"], type=int)
# args.add_argument("--plot", default=config["log"]["plot"], type=eval)
# args = args.parse_args()

# # Create an instance of your AGCRN model
# model = AGCRN(args)

# # Create some dummy input
# x = torch.randn(1, 10, 31, 1)
# targets = torch.randn(1, 1, 31, 1)

# # Generate a visualization of the forward pass
# dot = make_dot(model(x, targets), params=dict(model.named_parameters()))

# # Create a SummaryWriter
# writer = SummaryWriter("logs")

# # Save the graph to the SummaryWriter
# writer.add_graph(model, (x, targets))

# # Close the SummaryWriter
# writer.close()

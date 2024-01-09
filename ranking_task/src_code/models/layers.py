import torch
import torch.nn as nn

'''
This file is used only for KAR models.
'''

class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, fc_dims, input_dim, dropout):
        super(MLP, self).__init__()
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    """
    Mixture of Export
    """
    def __init__(self, moe_arch, inp_dim, dropout):
        super(MoE, self).__init__()
        export_num, export_arch = moe_arch
        self.export_num = export_num
        self.gate_net = nn.Linear(inp_dim, export_num)
        self.export_net = nn.ModuleList([MLP(export_arch, inp_dim, dropout) for _ in range(export_num)])

    def forward(self, x):
        gate = self.gate_net(x).view(-1, self.export_num)  # (bs, export_num)
        gate = nn.functional.softmax(gate, dim=-1).unsqueeze(dim=1) # (bs, 1, export_num)
        experts = [net(x) for net in self.export_net]
        experts = torch.stack(experts, dim=1)  # (bs, expert_num, emb)
        out = torch.matmul(gate, experts).squeeze(dim=1)
        return out

class HEA(nn.Module):
    """
    hybrid-expert adaptor
    """
    def __init__(self, ple_arch, inp_dim, dropout):
        super(HEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = ple_arch
        self.share_expt_net = nn.ModuleList([MLP(expt_arch, inp_dim, dropout) for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.ModuleList([nn.ModuleList([MLP(expt_arch, inp_dim, dropout)
                                                           for _ in range(spcf_expt_num)]) for _ in range(task_num)])
        self.gate_net = nn.ModuleList([nn.Linear(inp_dim, share_expt_num + spcf_expt_num)
                                   for _ in range(task_num)])

    def forward(self, x_list):
        gates = [net(x) for net, x in zip(self.gate_net, x_list)]
        gates = torch.stack(gates, dim=1)  # (bs, tower_num, expert_num), export_num = share_expt_num + spcf_expt_num
        gates = nn.functional.softmax(gates, dim=-1).unsqueeze(dim=2)  # (bs, tower_num, 1, expert_num)
        cat_x = torch.stack(x_list, dim=1)  # (bs, tower_num, inp_dim)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = torch.stack(share_experts, dim=2)  # (bs, tower_num, share_expt_num, E)
        spcf_experts = [torch.stack([net(x) for net in nets], dim=1)
                        for nets, x in zip(self.spcf_expt_net, x_list)]
        spcf_experts = torch.stack(spcf_experts, dim=1)  # (bs, tower_num, spcf_expt_num, num)
        experts = torch.cat([share_experts, spcf_experts], dim=2)  # (bs, tower_num, expert_num, E)
        export_mix = torch.matmul(gates, experts).squeeze(dim=2)  # (bs, tower_num, E)
        # print('export mix', export_mix.shape, 'tower num', self.tower_num)
        export_mix = torch.split(export_mix, dim=1, split_size_or_sections=1)
        out = [x.squeeze(dim=1) for x in export_mix]
        return out
    
class ConvertNet(nn.Module):
    """
    convert from semantic space to recommendation space
    """
    def __init__(self, export_num, specific_export_num, convert_arch, inp_dim, dropout, conv_type='HEA'):
        super(ConvertNet, self).__init__()
        self.type = conv_type
        print(self.type)
        if self.type == 'MoE':
            # print('convert module: MoE')
            moe_arch = export_num, convert_arch
            self.sub_module = MoE(moe_arch, inp_dim, dropout)
        elif self.type == 'HEA':
            # print('convert module: HEA')
            ple_arch = export_num, specific_export_num, convert_arch, 2
            self.sub_module = HEA(ple_arch, inp_dim, dropout)
        else:
            raise NotImplementedError

    def forward(self, x_list):
        if self.type == 'HEA':
            out = self.sub_module(x_list)
        else:
            out = [self.sub_module(x) for x in x_list]
        out = torch.cat(out, dim=-1)
        return out

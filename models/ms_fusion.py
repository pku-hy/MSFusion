import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class Multi_scale_fusion_Block(nn.Module):
    def __init__(self, configs):
        super(Multi_scale_fusion_Block,self).__init__()
        self.configs = configs
        self.down_sampling_window = configs.down_sampling_window
        self.seq_len = configs.seq_len
        self.down_sampling_layers = configs.down_sampling_layers
        
        
        self.residual_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    self.seq_len // (self.down_sampling_window ** (i+1)),
                    self.seq_len // (self.down_sampling_window ** i)
                          ),
                nn.GELU() ,
                nn.Dropout(0.1),
                # ConvResidualBlock(self.seq_len // (self.down_sampling_window ** i),self.seq_len // (self.down_sampling_window ** i)),
                # nn.Linear(
                #     self.seq_len // (self.down_sampling_window ** i),
                #     self.seq_len // (self.down_sampling_window ** i)
                #           ) ,
                # nn.GELU() ,
                # nn.Dropout(0.1),
            )
            for i in range(self.down_sampling_layers)
        ])
        
        self.res_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.seq_len // (self.down_sampling_window ** i)
                          ),
                nn.GELU() ,
                nn.Dropout(0.1),
                # nn.Linear(
                #     self.seq_len // (self.down_sampling_window ** i),
                #     self.seq_len // (self.down_sampling_window ** i)
                #           ) ,
                # nn.GELU() ,
                # nn.Dropout(0.1),
            )
            for i in range(self.down_sampling_layers+1)
        ])
        
    def forward(self,x_list):
        # first take direct decomposition
        # for i in range(1,len(x_list),1):
        #     x_h = x_list[i]
        #     bn , T , yi = x_h.size()
        #     # print(x_h.repeat(1,1,self.down_sampling_window).reshape(bn, -1 , yi).size())
        #     x_list[i-1] = x_list[i-1] - x_h.repeat(1,1,self.down_sampling_window).reshape(bn, -1 , yi)
        # take residual connect
        for i in range(len(x_list)-1,0,-1):
            x_list[i-1] = x_list[i-1] + self.residual_net[i-1](x_list[i].permute(0,2,1)).permute(0,2,1)
        # for i in range(len(x_list)):
        #     x_list[i] = x_list[i] + self.res_net[i](x_list[i].permute(0,2,1)).permute(0,2,1)
        return x_list

class Multi_scale_Prediction_Block(nn.Module):
    def __init__(self,configs):
        super(Multi_scale_Prediction_Block,self).__init__()
        self.configs = configs
        self.down_sampling_window = configs.down_sampling_window
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.predict_dropout = configs.predict_dropout
        self.down_sampling_layers = configs.down_sampling_layers
        
        self.predict_layers = nn.ModuleList()
        for i in range(self.down_sampling_layers+1):
            sl = self.seq_len // (self.down_sampling_window**i)
            pl = self.pred_len // (self.down_sampling_window**i)
            self.predict_layers.append(self._build_scale_predictor(sl,pl,self.configs.predict_layer))
            
    def _build_scale_predictor(self, seq_len, pred_len, num_layers):
        """
        Build the prediction layers for one temporal scale.
        """
        layers = []
        
        if num_layers <= 0:
            # Return an identity mapping when the layer count is nonpositive.
            return nn.Identity()
        
        # First layer: map seq_len to pred_len.
        layers.append(
            nn.Linear(seq_len, pred_len)
        )
        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.predict_dropout))
        
        # Intermediate layers preserve pred_len.
        for _ in range(num_layers - 2):
            layers.append(
                nn.Linear(pred_len, pred_len)
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.predict_dropout))
        
        # For multiple layers, use a final linear layer without activation or dropout.
        if num_layers >= 2:
            layers.append(
                nn.Linear(pred_len, pred_len)
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x_list):
        for i in range(len(x_list)):
            x_list[i] = x_list[i].permute(0,2,1)
            x_list[i] = self.predict_layers[i](x_list[i])
            x_list[i] = x_list[i].permute(0,2,1)
        return x_list

class MoE_fusion_Block(nn.Module):
    def __init__(self,configs):
        super(MoE_fusion_Block,self).__init__()
        self.configs = configs
        self.down_sampling_window = configs.down_sampling_window
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_layers = configs.down_sampling_layers
        self.residual_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    self.pred_len // (self.down_sampling_window ** (i+1)),
                    self.pred_len // (self.down_sampling_window ** i)
                        ),
                nn.GELU() ,
                nn.Dropout(self.configs.dropout),
            )
            for i in range(self.down_sampling_layers)
        ])
        
        # self.res_pred_layer = nn.Sequential(
        #         nn.Linear(
        #             self.pred_len ,
        #             self.pred_len
        #                   ),
        #         nn.GELU() ,
        #         nn.Dropout(self.configs.dropout),
        #     )

    def forward(self,x_list,weights=None):
        for i in range(len(x_list)-1,0,-1):
            x_list[i-1] = x_list[i-1] + self.residual_net[i-1](x_list[i].permute(0,2,1)).permute(0,2,1)
        # for i in range(len(x_list)-1,0,-1):
        #     x_list[0] = x_list[0] + self.residual_net[i-1](x_list[i].permute(0,2,1)).permute(0,2,1)
        # res_fusion = self.res_pred_layer(x_list[0].permute(0,2,1)).permute(0,2,1)  
        # x_list[0] = x_list[0] + res_fusion
        return x_list , weights

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window 
        self.c_in = configs.c_in

        self.enc_embeddings =  torch.nn.ModuleList(
            [
            DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, self.configs.predict_dropout)
            for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.c_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
        self.ms_fusion = Multi_scale_fusion_Block(self.configs)
        self.ms_predict = Multi_scale_Prediction_Block(self.configs)
        self.moe_fusion = MoE_fusion_Block(self.configs)
        
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.projection_layers = torch.nn.ModuleList(
            [
                nn.Linear(configs.d_model, 1, bias=True)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.c_in, out_channels=self.configs.c_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forward(self, x_enc, x_mark_enc ,stage):

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            if self.configs.use_norm:
                x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)
        
        # if stage != 1:
        #     x_list = self.ms_fusion(x_list)
        # embedding
        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embeddings[i](x, None)  # [B*N,T,C]
            enc_out_list.append(enc_out)
        
        # Future Multipredictor Mixing as decoder for future
        enc_out_list = self.ms_predict(enc_out_list)
        
        for i in range(self.configs.down_sampling_layers + 1):
            enc_out_list[i] = self.projection_layers[i](enc_out_list[i])
            enc_out_list[i] = enc_out_list[i].reshape(B, self.configs.c_out, self.pred_len//(self.down_sampling_window**i)).permute(0, 2, 1).contiguous() 

        if self.configs.use_norm:
            for i in range(self.configs.down_sampling_layers + 1):
                enc_out_list[i] = self.normalize_layers[i](enc_out_list[i], 'denorm')
            
        raw_predictions = list(enc_out_list)
        if stage != 1:
            enc_out_list, weights = self.moe_fusion(list(enc_out_list))
        return enc_out_list[0], raw_predictions

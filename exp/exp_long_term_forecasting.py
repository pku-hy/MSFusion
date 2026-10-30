from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=1e-2)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion =  nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion, stage=2):
        """
        stage:
            1 -> per-scale loss
            2 -> fusion loss
            3 -> fusion + scale regularization
        """
        total_loss = []
        self.model.eval()
        scale_loss = []
        for i in range(self.args.down_sampling_layers+1):
            scale_loss.append([])
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # if self.args.data in ['PEMS', 'Solar']:
                #     batch_x_mark = None
                #     batch_y_mark = None

                outputs_fused, outputs_list = self.model(batch_x, batch_x_mark, stage)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # outputs_fused = outputs_fused[:, -self.args.pred_len:, f_dim:]
                # outputs_list = [
                #     o[:, -self.args.pred_len:, f_dim:] for o in outputs_list
                # ]

                # ---------- multi-scale GT ----------
                y_list, _ = self.multi_scale_process_inputs(batch_y, None)

                # ---------- stage-wise loss ----------
                if stage == 1:
                    # only scale predictors
                    loss = 0.0
                    for i in range(self.args.down_sampling_layers+1):
                        scale_loss[i].append(criterion(outputs_list[i] , y_list[i].to(self.device)).item())
                        loss += criterion(outputs_list[i] , y_list[i].to(self.device))
                    loss = loss / len(outputs_list)

                elif stage == 2:
                    # only fusion
                    loss = criterion(outputs_fused, batch_y)

                elif stage == 3:
                    # fusion + light scale regularization
                    loss_fusion = criterion(outputs_fused, batch_y)

                    loss_scale = 0.0
                    for scale, (s_out, y) in enumerate(zip(outputs_list, y_list)):
                        loss_scale += 2.0 ** (-scale) * criterion(s_out, y.to(self.device))
                    loss_scale = loss_scale / len(outputs_list)

                    alpha = self.args.scale_reg_weight
                    loss = loss_fusion + alpha * loss_scale

                else:
                    raise ValueError(f'Unknown stage: {stage}')

                total_loss.append(loss.item())
        scale_print = f""
        if stage==1:
            for i in range(len(scale_loss)):
                scale_print = scale_print + f"scale{i} loss :{float(np.mean(scale_loss[i])):.6f}"
            print('Val',scale_print)
        self.model.train()
        return float(np.mean(total_loss))

    def train_stagewise(self, setting):
        # get datasets / loaders (same as before)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # --- prepare ---
        criterion = self._select_criterion()

        # keywords for identifying params: allow user to set or use defaults
        scale_keywords = getattr(self.args, 'scale_key_words', ['projection', 'predict', 'embedding', 'normalize'])
        fusion_keywords = getattr(self.args, 'fusion_key_words', ['moe_fusion'])

        # Stage epoch counts (fallback to full epochs split)
        s1_epochs = getattr(self.args, 'stage1_epochs', max(1, self.args.train_epochs // 3))
        s2_epochs = getattr(self.args, 'stage2_epochs', max(1, self.args.train_epochs // 3))
        s3_epochs = getattr(self.args, 'stage3_epochs', max(1, self.args.train_epochs - s1_epochs - s2_epochs))

        print(f"Stage-wise training epochs: S1={s1_epochs}, S2={s2_epochs}, S3={s3_epochs}")
        print("=== Stage 1: train scale predictors (freeze fusion) ===")

        self._freeze_all(True)
        self._set_requires_grad_by_keywords(scale_keywords, True)

        model_optim = self._build_optimizer_with_param_groups(
            base_lr=self.args.learning_rate,
            fusion_keywords=fusion_keywords,
            scale_keywords=scale_keywords,
            fusion_lr_mul=0.0,
            scale_lr_mul=1.0,
            weight_decay=1e-2
        )

        best_val = float("inf")
        wait = 0
        patience = getattr(self.args, "stage1_patience", 5)
        best_path_s1 = os.path.join(path, "checkpoint_stage1_best.pth")

        for epoch in range(s1_epochs):
            self.model.train()
            epoch_losses = []
            scale_loss = []
            for i in range(self.args.down_sampling_layers+1):
                scale_loss.append([])
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = None # batch_x_mark.float().to(self.device)

                outputs, outputs_list = self.model(batch_x, batch_x_mark, 1)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                y_list, _ = self.multi_scale_process_inputs(batch_y_cut, None)

                loss = 0.0
                for s_idx, s_out in enumerate(outputs_list):
                    scale_loss[s_idx].append(criterion(s_out, y_list[s_idx].to(self.device)).item())
                    loss += criterion(s_out, y_list[s_idx].to(self.device))
                loss /= len(outputs_list)

                loss.backward()
                model_optim.step()
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))
            scale_print = f""
            for i in range(len(scale_loss)):
                scale_print = scale_print + f"scale{i} loss: {float(np.mean(scale_loss[i])):.6f} "
            print('Train',scale_print)
            val_loss = self.vali(vali_data, vali_loader, criterion,1)
            print(f"[Stage1] Epoch {epoch+1} | train {train_loss:.6f} | val {val_loss:.6f}")
            
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), best_path_s1)
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Stage1] Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(torch.load(best_path_s1, map_location='cpu'))
        self.model.to(self.device)
        print("[Stage1] Loaded best checkpoint")

        # -------- Stage 2: freeze predictors, train fusion ----------
        print("=== Stage 2: train fusion (freeze scale predictors) ===")

        self._freeze_all(True)
        self._set_requires_grad_by_keywords(fusion_keywords, True)

        model_optim = self._build_optimizer_with_param_groups(
            base_lr=self.args.learning_rate,
            fusion_keywords=fusion_keywords,
            scale_keywords=scale_keywords,
            fusion_lr_mul=1.0,
            scale_lr_mul=0,
            weight_decay=1e-2
        )

        best_val = float("inf")
        wait = 0
        patience = getattr(self.args, "stage2_patience", 3)
        best_path_s2 = os.path.join(path, "checkpoint_stage2_best.pth")

        for epoch in range(s2_epochs):
            self.model.train()
            epoch_losses = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark =  None #batch_x_mark.float().to(self.device)

                outputs, _ = self.model(batch_x, batch_x_mark,2)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y_cut)
                loss.backward()
                model_optim.step()
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))
            val_loss = self.vali(vali_data, vali_loader, criterion,2)
            print(f"[Stage2] Epoch {epoch+1} | train {train_loss:.6f} | val {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), best_path_s2)
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Stage2] Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(torch.load(best_path_s2, map_location='cpu'))
        self.model.to(self.device)
        print("[Stage2] Loaded best checkpoint")


        print("=== Stage 3: light joint fine-tuning (constrained) ===")

        # 1. unfreeze all
        self._freeze_all(False)

        # 2. optimizer with LR multipliers
        model_optim = self._build_optimizer_with_param_groups(
            base_lr=self.args.learning_rate,
            fusion_keywords=fusion_keywords,
            scale_keywords=scale_keywords,
            fusion_lr_mul=getattr(self.args, "fusion_lr_mul", 0.1),
            scale_lr_mul=getattr(self.args, "scale_lr_mul", 0.01),   
            weight_decay=1e-2
        )

        best_val = float("inf")
        wait = 0
        patience = getattr(self.args, "stage3_patience", 3)
        best_path_s3 = os.path.join(path, "checkpoint_stage3_best.pth")

        for epoch in range(s3_epochs):
            self.model.train()
            epoch_losses = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark =  None #batch_x_mark.float().to(self.device)
                l_time = time.time()
                outputs, outputs_list = self.model(batch_x, batch_x_mark,3)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # === 主损失（融合结果） ===
                loss_main = criterion(outputs, batch_y_cut)

                # === 约束损失：保持各尺度预测不退化 ===
                alpha = self.args.scale_reg_weight
                y_list, _ = self.multi_scale_process_inputs(batch_y_cut, None)
                loss_reg = 0.0
                
                for s_idx, s_out in enumerate(outputs_list):
                    loss_reg += (1/2**s_idx) * criterion(s_out, y_list[s_idx].to(self.device))
                
                loss_reg /= len(outputs_list)

                loss = loss_main + alpha * loss_reg
                loss.backward()
                model_optim.step()
                #print("Iter time: {}".format(time.time() - l_time))    
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))
            val_loss = self.vali(vali_data, vali_loader, criterion,3)
            print(f"[Stage3] Epoch {epoch+1} | train {train_loss:.6f} | val {val_loss:.6f} ")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), best_path_s3)
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Stage3] Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(torch.load(best_path_s3, map_location='cpu'))
        self.model.to(self.device)
        print("[Stage3] Loaded best checkpoint")
        return self.model



        
        # print(f"Saved multi-scale visualization to: {save_path}")

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        for stage in [1,2,3]:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, f'checkpoint_stage{stage}_best.pth'), map_location='cpu')
            )
            self.model.to(self.device)

            preds = []
            trues = []

            folder_path = rf'./test_results/{setting}/stage{stage}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    if self.args.data in ['PEMS', 'Solar']:
                        batch_x_mark = None
                        batch_y_mark = None

                    # ---------- forward ----------
                    outputs_fused, outputs_list = self.model(batch_x, batch_x_mark, stage)
                    f_dim = 0
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    outputs_fused = outputs_fused[:, -self.args.pred_len:, f_dim:]

                    pred = outputs_fused
                    # ---------- numpy ----------
                    pred = pred.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                    preds.append(pred)
                    trues.append(true)

                    # ---------- visualization ----------
                    if self.args.plot and i % 20 == 0:
                        input_x = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input_x.shape
                            input_x = test_data.inverse_transform(
                                input_x.squeeze(0)
                            ).reshape(shape)

                        gt = np.concatenate(
                            (input_x[0, :, -1], true[0, :, -1]), axis=0
                        )
                        pd = np.concatenate(
                            (input_x[0, :, -1], pred[0, :, -1]), axis=0
                        )
                        visual(gt, pd, os.path.join(folder_path,rf'{str(i)}.pdf'))

            # ---------- metric ----------
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f'stage={stage} mse:{mse}, mae:{mae}')
            import json
            with open(os.path.join(folder_path, 'metrics.json'), 'w') as f:
                json.dump({'stage': stage, 'mse': float(mse), 'mae': float(mae)}, f, indent=2)
    
        return
    
    def multi_scale_process_inputs(self, x, x_mark):
        if self.args.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.args.down_sampling_window, return_indices=False)
        elif self.args.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.args.down_sampling_window)
        elif self.args.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.args.c_in, out_channels=self.args.c_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.args.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x, x_mark
        # B,T,C -> B,C,T
        x = x.permute(0, 2, 1)

        x_ori = x
        x_mark_mark_ori = x_mark

        x_sampling_list = []
        x_mark_sampling_list = []
        x_sampling_list.append(x.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark)

        for i in range(self.args.down_sampling_layers):
            x_sampling = down_pool(x_ori)

            x_sampling_list.append(x_sampling.permute(0, 2, 1))
            x_ori = x_sampling

            if x_mark_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_mark_ori[:, ::self.args.down_sampling_window, :])
                x_mark_mark_ori = x_mark_mark_ori[:, ::self.args.down_sampling_window, :]

        x = x_sampling_list
        if x_mark_mark_ori is not None:
            x_mark = x_mark_sampling_list
        else:
            x_mark = x_mark

        return x, x_mark

        # ---------------- helper utilities ----------------
    
    def _param_is_in_keywords(self, name, keywords):
        # normalize for DataParallel 'module.' prefix
        if name.startswith('module.'):
            name = name[len('module.'):]
        for k in keywords:
            if k in name:
                return True
        return False

    def _set_requires_grad_by_keywords(self, keywords, requires_grad):
        """
        Set requires_grad for parameters whose name contains any keyword in `keywords`.
        If keywords is empty list -> do nothing.
        """
        if not keywords:
            return
        for name, p in self.model.named_parameters():
            if self._param_is_in_keywords(name, keywords):
                p.requires_grad = requires_grad

    def _freeze_all(self, freeze=True):
        for p in self.model.parameters():
            p.requires_grad = not freeze

    def _build_optimizer_with_param_groups(self, base_lr, fusion_keywords, scale_keywords, fusion_lr_mul=1.0, scale_lr_mul=0.1, weight_decay=1e-2):
        """
        Create optimizer with param groups:
          - fusion params get base_lr * fusion_lr_mul
          - scale params get base_lr * scale_lr_mul
          - others get base_lr * 0.01 (small)
        """
        fusion_params = []
        scale_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            # print(name)
            if not p.requires_grad:
                continue
            if self._param_is_in_keywords(name, fusion_keywords):
                fusion_params.append(p)
            elif self._param_is_in_keywords(name, scale_keywords):
                scale_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': base_lr * fusion_lr_mul})
        if scale_params:
            param_groups.append({'params': scale_params, 'lr': base_lr * scale_lr_mul})
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr * 0.01})
        optim = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
        return optim
    # --------------------------------------------------

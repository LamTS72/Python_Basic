import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
import logging
import sys
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
import copy
from sklearn.metrics import classification_report

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.supported_metrics = {
            "accuracy": {
                "func": metrics.accuracy_score,
                "params": {}
            },
            "f1_score": {
                "func": metrics.f1_score,
                "params": {"average": "macro"}
            },
            "precision": {
                "func": metrics.precision_score,
                "params": {"average": "macro"}
            },
            "recall": {
                "func": metrics.recall_score,
                "params": {"average": "macro"}
            }
        }
        self.device = torch.device("cpu") #default
        self.pbar = None
        self.metrics = None 
        self.optimizer = None
        self.loss_meter = None
        self.key_metric = None
        self.best_perf = float("inf") #assume: optimization = minimize key_metric
        self.save_best_to = None
        self.save_last_to = None
        self.train_prefix = "Training summary"
        self.valid_prefix = "Validation summary"

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def _verify_metrics(self, metrics):
        if metrics is None:
            self.key_metric = "f1_score"
            self.metrics = ["f1_score"]
        else:
            self.metrics = []
            for metric in metrics:
                if metric.lower() in self.supported_metrics:
                    self.metrics.append(metric.lower())
                else:
                    raise Exception(f"Metric '{metric}': not supported.")
            self.key_metric= self.metrics[0]
                
        
    def compile(self, optimizer, loss_meter, metrics=["f1_score"], 
                save_best_to="the_best.pt",
                save_last_to="last_model.pt"
               ):
        self.optimizer = optimizer
        self.loss_meter = loss_meter
        self.save_best_to = save_best_to
        self.save_last_to = save_last_to
        self._verify_metrics(metrics)
        
    ####################################################################################
    ### Training
    ####################################################################################
    def fit(self, train_loader, valid_loader, nepoches):
        """
        """
        self._on_train_start(train_loader, valid_loader, nepoches)

        nbatches = len(train_loader)
        self.nepoches, self.nbatches = nepoches, nbatches
        epoch_str, batch_str = str(nepoches), str(nbatches)
        nsteps = nepoches*nbatches
        self.pbar = tqdm(range(nsteps), total=nsteps, desc="training", position=0, leave=True)

        for epoch in range(nepoches):
            nsamples = 0
            self._on_train_epoch_start()
            for bidx, batch in enumerate(train_loader):
                self._on_train_batch_start()
                
                #SGD's steps:
                batch_tokens, batch_masks, batch_labels = [item.to(self.device) for item in batch]
                nsamples +=  batch_tokens.size(0)
                # SGD::forward
                self.zero_grad()        
                batch_preds = self.forward(batch_tokens, batch_masks)
                loss = self.loss_meter(batch_preds, batch_labels)
                # SGD::backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                # SGD::update
                self.optimizer.step()

                # record loss, preds and labels for computing metrics
                self.total_loss = self.total_loss + loss.item()
                batch_preds = F.softmax(batch_preds)
                batch_preds = batch_preds.detach().cpu().numpy()
                batch_preds = np.argmax(batch_preds, axis=-1)
                
                # append the model predictions
                self.total_preds += batch_preds.tolist()
                self.total_labels += batch_labels.tolist()

                # update pbar's description
                msg = "Epoch: {:>{w1}d}/{:s} | Batch: {:>{w2}d}/{:s}".format(epoch + 1, epoch_str, 
                                                                             bidx + 1, batch_str,
                                                                             w1=len(epoch_str), w2=len(batch_str))
                self.pbar.set_description(msg, refresh=True)
                self.pbar.update(1)
                sys.stdout.flush()
                
                #end: batch
                self._on_train_batch_end()
            self._on_train_epoch_end(epoch + 1, bidx, nsamples)
        self.pbar.close()
        self._on_train_end()
        
    def _on_train_start(self, train_loader, valid_loader, nepoches):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.nepoches = nepoches
        self.train() # to train mode
        #
        self.total_loss, self.total_accuracy = 0, 0
        self.total_preds, self.total_labels = [], []
        self.best_perf = float("inf")
        
    def _on_train_end(self):
        # save
        self.save(self.save_last_to)
        
    def _on_train_epoch_start(self):
        self.total_loss, self.total_accuracy = 0, 0
        self.total_preds, self.total_labels = [], []
        self.train() # to train mode

    def _desc_metrics(self, total_labels, total_preds, width=15, dec=4):
        metrics_msg = []
        scores = []
        for name in self.metrics:
            metric_func = self.supported_metrics[name]["func"]
            metric_params = self.supported_metrics[name]["params"]
            score = metric_func(total_labels, total_preds, **metric_params)
            scores.append(score)
            desc = "{:>{w}.{d}f}".format(score, w=width, d=dec)
            metrics_msg.append(desc)
        metrics_msg = "".join(metrics_msg)
        return metrics_msg, scores
        
    def _on_train_epoch_end_BCK(self, epoch, bidx, nsamples):
        header =  "{:<20s}".format(' ')
        header += "{:>20s}".format("Loss(per-sample)")
        header += "".join(["{:>15s}".format(m) for m in self.metrics])
        
        metrics_msg = []
        for name in self.metrics:
            metric_func = self.supported_metrics[name]["func"]
            metric_params = self.supported_metrics[name]["params"]
            score = metric_func(self.total_labels, self.total_preds, **metric_params)
            desc = "{:>15.4f}".format(score)
            metrics_msg.append(desc)
        metrics_msg = "".join(metrics_msg)
        
        line =  "{:<20s}".format(self.train_prefix)
        line += "{:>20.4f}".format(self.total_loss/nsamples)
        line += metrics_msg
        
        print(header)
        print(line)
        sys.stdout.flush()
        #
        self._validate(self.valid_loader)

    def _on_train_epoch_end(self, epoch, bidx, nsamples):
        header =  "{:<20s}".format(' ')
        header += "{:>20s}".format("Loss(per-sample)")
        header += "".join(["{:>15s}".format(m) for m in self.metrics])
        
        metrics_msg, _ = self._desc_metrics(self.total_labels, self.total_preds)
        line =  "{:<20s}".format(self.train_prefix)
        line += "{:>20.4f}".format(self.total_loss/nsamples)
        line += metrics_msg
        
        print(header)
        print(line)
        sys.stdout.flush()
        #
        self._validate(self.valid_loader)

    def _on_train_batch_start(self):
        pass

    def _on_train_batch_end(self):
        pass

    ####################################################################################
    ### Validation
    ####################################################################################
    def _on_validate_start(self, valid_loader):
        self.eval() # to eval mode

    def _on_validate_end_BCK(self, nsamples, valid_total_loss, valid_total_preds, valid_total_labels, valid_loader):
        metrics_msg = []
        scores = []
        for name in self.metrics:
            metric_func = self.supported_metrics[name]["func"]
            metric_params = self.supported_metrics[name]["params"]
            score = metric_func(valid_total_labels, valid_total_preds, **metric_params)
            scores.append(score)
            desc = "{:>15.4f}".format(score)
            metrics_msg.append(desc)
        metrics_msg = "".join(metrics_msg)
        
        line =  "{:<20s}".format(self.valid_prefix)
        line += "{:>20.4f}".format(valid_total_loss/nsamples)
        line += metrics_msg
        
        print(line)
        sys.stdout.flush()
        
        # save model
        key_score = scores[0]
        if key_score < self.best_perf:
            self.best_perf = key_score
            # save
            base_state = {
                'nepoches': self.nepoches,
                'optimizer': self.optimizer,
                'loss_meter': self.loss_meter,
                'metrics': self.metrics,
                'model': self.state_dict(),
            }
            torch.save(base_state, self.save_best_to)

    def _on_validate_end(self, nsamples, total_loss, total_preds, total_labels, valid_loader):
        metrics_msg, scores = self._desc_metrics(total_labels, total_preds)
        line =  "{:<20s}".format(self.valid_prefix)
        line += "{:>20.4f}".format(total_loss/nsamples)
        line += metrics_msg
        
        print(line)
        sys.stdout.flush()
        
        # save model
        key_score = scores[0]
        if key_score < self.best_perf:
            self.best_perf = key_score
            self.save(self.save_best_to)
            
    def save(self, filename):
        # save
        state = {
            'nepoches': self.nepoches,
            'optimizer': self.optimizer,
            'loss_meter': self.loss_meter,
            'metrics': self.metrics,
            'model': self.state_dict(),
        }
        torch.save(state, filename)

    def _forward_a_loader(self, loader):
        total_loss = 0
        total_preds = []
        total_labels = []
        nsamples = 0
        for bidx, batch in enumerate(loader):
            batch_tokens, batch_masks, batch_labels = [item.to(self.device) for item in batch]
            nsamples +=  batch_tokens.size(0)

            with torch.no_grad():
                batch_preds = self.forward(batch_tokens, batch_masks)
                loss = self.loss_meter(batch_preds, batch_labels)
            
                # record preds and labels for computing metrics
                batch_preds = F.softmax(batch_preds)
                batch_preds = batch_preds.detach().cpu().numpy()
                batch_preds = np.argmax(batch_preds, axis=-1)
                
                total_loss += loss.item()
                total_preds += batch_preds.tolist()
                total_labels += batch_labels.tolist()
        return nsamples, total_loss, total_preds, total_labels

    def _validate(self, valid_loader):
        self._on_validate_start(valid_loader)
        self._on_validate_end(*self._forward_a_loader(valid_loader), valid_loader)

    ####################################################################################
    ### Evaluation
    ####################################################################################
    def _on_evaluate_start(self, valid_loader):
        self.eval() # to eval mode

    def _on_evaluate_end(self, nsamples, total_loss, total_preds, total_labels, loader):
        print("="*100)
        print("Evaluation: ")
        print("-"*100)
        print(classification_report(total_labels, total_preds))
        print("="*100)
        print("Loss(per-sample):{:15.4f}".format(total_loss/nsamples))
        print("="*100)
        return nsamples, total_loss, total_preds, total_labels
        
    
    def evaluate(self, test_loader):
        self._on_evaluate_start(test_loader)
        return self._on_evaluate_end(*self._forward_a_loader(test_loader), test_loader)

        
        
    
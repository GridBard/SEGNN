#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:16:28 2021

@author: Gong Dongsheng
"""

import torch
import sys

from sklearn.metrics import classification_report, matthews_corrcoef, cohen_kappa_score, confusion_matrix

def evaluation(output, y_test):
    y_predict = output.argmax(1)
    evalue = classification_report(y_test.cpu(), y_predict.cpu(), output_dict=True, zero_division=1)
    kappa = cohen_kappa_score(y_test.cpu(), y_predict.cpu())
    mcc = matthews_corrcoef(y_test.cpu(), y_predict.cpu())
    return evalue, kappa, mcc

def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
    loss = org_loss_func(preds, labels)
    #_mask = torch.FloatTensor(mask)
    #_mask /= _mask.sum()
    _mask =mask/mask.sum()
    loss *= _mask
    return loss.sum()


def masked_accuracy(preds, labels, mask):
    acc = torch.eq(preds.argmax(1), labels).float()
    #_mask = torch.FloatTensor(mask)
    _mask =mask/mask.sum()
    acc *= _mask
    return acc.sum().item()

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss

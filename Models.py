# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:35:43 2019

@author: Zhou
"""

from DecoderWrappers import SampleDecodingWrapper, BeamSearchWrapper
from Utils import save, batch_bleu, batch_meteor, batch_rouge
from Data import id2word
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn

beam_width = 5
n_best = 1
sampling_temp = 1.
sampling_topk = -1
max_iter = 30
length_penalty = 1.
coverage_penalty = 0.


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_batch, tgt_batch):
        tgt_batch = tgt_batch[:, :-1]
        src_lengths = src_batch[0].ptr[1:] - src_batch[0].ptr[:-1] + src_batch[5]
        memory, final_state = self.encoder(*src_batch)
        logits, attn_history = self.decoder(tgt_batch, final_state, memory, src_lengths)
        return logits, attn_history


class Translator(object):
    def __init__(self, model, sampling_temp=sampling_temp, sampling_topk=sampling_topk,
                 beam_width=beam_width, n_best=n_best, max_iter=max_iter,
                 length_penalty=length_penalty, coverage_penalty=coverage_penalty,
                 metrics=['bleu'], unk_replace=False, smooth=3):
        self.model = model
        self.metrics = metrics
        self.unk_replace = unk_replace
        self.smooth = smooth

        if not beam_width or beam_width == 1:
            self.wrapped_decoder = SampleDecodingWrapper(
                model.decoder, sampling_temp, sampling_topk, max_iter)
        else:
            self.wrapped_decoder = BeamSearchWrapper(
                model.decoder, beam_width, n_best, max_iter, length_penalty, coverage_penalty)

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        metrics = set(metrics)
        all_metrics = {'bleu', 'rouge', 'meteor'}
        if not metrics.issubset(all_metrics):
            raise ValueError('Unkown metric(s): ' + str(metrics.difference(all_metrics)))
        self._metrics = metrics

    def translate_batch(self, src_batch, raw_tgt=None):
        reports = dict(scores=None, attn_history=None)
        with torch.no_grad():
            src_lengths = src_batch[0].ptr[1:] - src_batch[0].ptr[:-1] + src_batch[5]
            memory, final_state = self.model.encoder(*src_batch)
            predicts, reports['scores'], reports['attn_history'] = \
                self.wrapped_decoder(final_state, memory, src_lengths)

        if type(self.wrapped_decoder) is BeamSearchWrapper:
            predicts = [b[0] for b in predicts]
            reports['scores'] = [b[0] for b in reports['scores']]
            if reports['attn_history'][0]:
                reports['attn_history'] = [b[0] for b in reports['attn_history']]

        predicts = id2word(predicts, self.model.decoder.field)
        if 'bleu' in self._metrics:
            reports['bleu'] = batch_bleu(predicts, raw_tgt, self.smooth) * 100
        predicts = [' '.join(s) for s in predicts]

        if not self._metrics.isdisjoint({'rouge', 'meteor'}):
            targets = [' '.join(s) for s in raw_tgt]
            if 'rouge' in self._metrics:
                rouge = batch_rouge(predicts, targets)
                reports['rouge'] = rouge['rouge-l']['f'] * 100
            if 'meteor' in self._metrics:
                reports['meteor'] = batch_meteor(predicts, targets) * 100

        return predicts, reports

    def __call__(self, batches, save_path=None):
        self.model.eval()
        results = []
        reports = defaultdict(float, scores=[], attn_history=[])

        pbar = tqdm(batches, desc='Translating...')
        for batch in pbar:
            predicts, reports_ = self.translate_batch(*batch)
            pbar.set_postfix({metric: reports_[metric] for metric in self._metrics})
            results.extend(predicts)
            for metric in self._metrics:
                reports[metric] += reports_[metric]
        #            reports['scores'].extend(reports_['scores'])
        # reports['attn_history'].extend(reports_['attn_history'])

        for metric in self._metrics:
            reports[metric] /= len(batches)
            print('total {}: {:.2f}'.format(metric, reports[metric]))
        if save_path is not None:
            save(results, save_path)
        return results, reports

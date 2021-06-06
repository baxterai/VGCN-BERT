# -*- coding: utf-8 -*-

# # # #
# model_vanilla_vgcn_bert_mlm.py
# @author Zhibin.LU
# @created ?
# @last-modified ?
# @website: https://louis-udm.github.io
# @description: Models
# # # #

"""model_vanilla_vgcn_bert_mlm.py

# Requirements:
Python 3.7 and PyTorch 1.0

Anaconda installation instructions:
conda create -n vgcnbert python=3.7
source activate vgcnbert
conda install -c pytorch pytorch=1.0
python -m pip install --upgrade pip --user
pip install nltk==3.4
pip install pytorch-pretrained-bert==0.6.2 
pip install scipy==1.4
pip install sklearn
pip install pandas

# License:
MIT License

# Usage:
see train_vanilla_vgcn_bert_mlm.py

# Requirements:

# Description:

- Additional Authors: Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

"""

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
import copy

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from model_vgcn_bert import VocabGraphConvolution

bertFinalLayerEmbeddingDimesion = 768	#https://www.statestitle.com/resource/understanding-berts-semantic-interpretations/
#bert final layer dimensions: 768
#gcn final layer dimensions: 768

debugTestPerformanceOnlyUseBertEmbeddings = False	#for benchmarking only

#BertForMaskedLM (hugging face 0.6.2) documentation:
#https://github.com/huggingface/transformers/tree/b832d5bb8a6dfc5965015b828e577677eace601e#overview
#https://github.com/huggingface/transformers/blob/b832d5bb8a6dfc5965015b828e577677eace601e/pytorch_pretrained_bert/modeling.py#L793
#description: This module comprises the BERT model followed by the masked language modeling head
	
class Vanilla_VGCN_Bert_mlm(BertForMaskedLM):

	def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, num_labels, output_attentions=False, keep_multihead_output=False):
		super(Vanilla_VGCN_Bert_mlm, self).__init__(config)	#.__init__(config, num_labels)
		
		self.vocab_gcn=VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256
	
		#inherited from BertForMaskedLM (hugging face 0.6.2):
		#implied: self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
		#implied: self.bert = BertModel(config) 
			#note from class BertModel(BertPreTrainedModel):
				#def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
				#Outputs: Tuple of (encoded_layers, pooled_output)
        			#`encoded_layers`: controled by `output_all_encoded_layers` argument:
						#output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding to the last attention block of shape [batch_size, sequence_length, hidden_size],
		
		self.linearIntegration = nn.Linear((gcn_embedding_dim+1)*bertFinalLayerEmbeddingDimesion, bertFinalLayerEmbeddingDimesion)	#CHECKTHIS: VGCN-Bert integration method
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.mask_token_id = None	
		self.ignore_index = None	
		

	def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, maskIndex=None):	#maskIndex=None	#BertForSequenceClassification: attention_mask=None, output_all_encoded_layers=False, head_mask=None
				
		words_embeddings = self.bert.embeddings.word_embeddings(input_ids)
		
		vocab_input=gcn_swop_eye.matmul(words_embeddings).transpose(1,2)
		gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input).transpose(1,2)	#shape: [8, 768]
		 
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)	#shape: [batch_size, sequence_length, hidden_size]
		
		sequence_length = sequence_output.shape[1]
		sequence_outputAtMaskIndex = sequence_output[:, maskIndex, :]	#shape: [batch_size, 1, hidden_size]
	
		#note Vanilla_VGCN_Bert_mlm(compared to BertForMaskedLM) returns data for sequenceIndex==maskIndex instead of entire sequence_length (as this is sufficient for prediction_scores and loss) 
			
		if(debugTestPerformanceOnlyUseBertEmbeddings):
			#approximates default BertForMaskedLM behaviour:
			prediction_scores = self.cls(sequence_output)	#temp for dimensions print test
			prediction_scoresAtMaskIndex = prediction_scores[:, maskIndex, :]
			prediction_scoresAtMaskIndex = torch.unsqueeze(prediction_scoresAtMaskIndex, 1)	#shape: [batch_size, 1, vocab_size]
		else:
			#add a linear layer between sequence_output and cls (masked language modeling head), which integrates the gcn vocab output embeddings:
			cat_out=torch.cat((gcn_vocab_out.squeeze(1),sequence_outputAtMaskIndex), dim=1)		
			cat_out = self.dropout(cat_out)
			integratedOutput = self.linearIntegration(cat_out)
			integratedOutput = torch.unsqueeze(integratedOutput, 1)
			prediction_scoresAtMaskIndex = self.cls(integratedOutput)	#shape: [batch_size, 1, vocab_size]
	
		if masked_lm_labels is not None:
			masked_lm_loss = self.calculateLoss(prediction_scoresAtMaskIndex, masked_lm_labels, maskIndex)
			return masked_lm_loss
		else:
			#return prediction_scores
			return prediction_scoresAtMaskIndex
			
		#`masked_lm_loss`: masked language modeling loss.  of shape [].
		#`prediction_scores`: masked language modeling logits of shape [batch_size, sequence_length, vocab_size].
		#`prediction_scoresAtMaskIndex`: masked language modeling logits of shape [batch_size, 1, vocab_size].		
	
	def initialiseTokenIds(self, mask_token_id, ignore_token_id):	#additional initialisation function for class Vanilla_VGCN_Bert_mlm
		self.mask_token_id = mask_token_id	#derivation requires tokenizer object from train_vanilla_vgcn_bert_mlm
		self.ignore_token_id = ignore_token_id	#derivation requires tokenizer object from train_vanilla_vgcn_bert_mlm
	
	def calculateLoss(self, prediction_scoresAtMaskIndex, masked_lm_labels, maskIndex):
		ignore_index = self.ignore_index
		masked_lm_labelsAtMaskIndex = masked_lm_labels[:, maskIndex]
		masked_lm_labelsAtMaskIndex = torch.unsqueeze(masked_lm_labelsAtMaskIndex, 1)	#shape: [batch_size, 1, vocab_size]
		loss_fct = CrossEntropyLoss(ignore_index=-1)
		masked_lm_loss = loss_fct(prediction_scoresAtMaskIndex.view(-1, self.config.vocab_size), masked_lm_labelsAtMaskIndex.view(-1))
		#masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))		
		return masked_lm_loss

	def generateMaskedInputIdsAndLabels(self, input_ids, maskIndex):
		mask_token_id = self.mask_token_id
		ignore_token_id = self.ignore_token_id
		masked_input_ids = copy.deepcopy(input_ids)
		masked_lm_labels = copy.deepcopy(input_ids)
		masked_input_ids[:, maskIndex] = mask_token_id
		masked_lm_labels[masked_input_ids != mask_token_id] = ignore_token_id
		return masked_input_ids, masked_lm_labels

	def calculatePredictionAccuracy(self, input_ids, prediction_ids):
		predictionLabels = copy.deepcopy(input_ids)
		eval_accuracy = torch.eq(prediction_ids, predictionLabels).sum().item()
		return eval_accuracy

	def calculatePredictionAccuracyAtMaskIndex(self, input_id, prediction_id):
		return self.calculatePredictionAccuracy(input_id, prediction_id)

	def calculateBestPrediction(self, logits, maskIndex):
		predictionProbabilities = logits[:, maskIndex]
		return self.calculateBestPredictionAtMaskIndex(predictionProbabilities)

	def calculateBestPredictionAtMaskIndex(self, prediction_scoresAtMaskIndex):
		predictionProbabilities = prediction_scoresAtMaskIndex
		topPredictionsNumber = 1
		predictionProbabilitiesTopTensor = predictionProbabilities.topk(topPredictionsNumber)
		predictionProbabilitiesTopIndices = predictionProbabilitiesTopTensor[1]	#second element #https://pytorch.org/docs/1.0.0/torch.html#torch.topk
		#predictionProbabilitiesTopIndices = predictionProbabilitiesTopTensor.indices	#pytorch 1.8	#https://pytorch.org/docs/stable/generated/torch.topk.html
		return predictionProbabilitiesTopIndices
	
	

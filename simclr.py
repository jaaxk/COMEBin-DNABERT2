# modified from https://github.com/sthalles/SimCLR
import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, load_checkpoint
import pandas as pd
import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        """
        Initialize the SimCLR model and related components.

        :param kwargs: Keyword arguments including 'args', 'model', 'optimizer', 'scheduler'.
        """
        self.args = kwargs['args']
        torch.set_num_threads(self.args.num_threads)
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.output_path)
        logging.basicConfig(filename=os.path.join(self.args.output_path, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        """
        Calculate the InfoNCE loss for SimCLR.

        :param features: Input features.
        :return: Logits and labels for the loss.
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(-1, 1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.args.n_views - 1, -1).flatten(0, 1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def covmodel_info_nce_loss(self, features):
        """
        Calculate the InfoNCE loss for the coverage model.

        :param features: Input features.
        :return: Logits and labels for the loss.
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(-1, 1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.args.n_views - 1, -1).flatten(0, 1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.covmodel_temperature
        return logits, labels

    def kmermodel_info_nce_loss(self, features):
        """
        Calculate the InfoNCE loss for the k-mer model.

        :param features: Input features.
        :return: Logits and labels for the loss.
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(-1, 1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives = negatives[:, None].expand(-1, self.args.n_views - 1, -1).flatten(0, 1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.kmermodel_temperature
        return logits, labels


    def train(self, train_loader, data, namelist):
        """
        Train the SimCLR model.

        :param train_loader: Data loader for training.
        :param data: Input data.
        :param namelist: List of sequence names.
        """
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.output_path, self.args)

        earlystop_epoch=0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        # logging.info(f"Training with cpu: {self.args.disable_cuda}.")
        
        best_acc = 0.0

        for epoch_counter in range(self.args.epochs):
            # modified
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            num_batches = 0

            for contig_features in tqdm(train_loader):
                contig_features = torch.cat(contig_features, dim=0)

                contig_features = contig_features.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(contig_features)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                epoch_accuracy += top1[0] #.item()?
                epoch_loss += loss #.item()
                num_batches += 1    

            #top1, top5 = accuracy(logits, labels, topk=(1, 5)) #this line was originally in this loop (mistake?)
            #Using average epoch loss and accuracy instead
            avg_accuracy = epoch_accuracy / num_batches
            avg_loss = epoch_loss / num_batches
            is_best = avg_accuracy > best_acc
            if is_best:
                best_acc = avg_accuracy
                print(f'Best accuracy: {best_acc} at epoch {epoch_counter}.')
            if not self.args.notuse_scheduler:
                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
            #logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}") #We should probably use epoch-level accuracy and loss instead:
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss:.4f}\tAccuracy: {avg_accuracy:.2f}")
            print(f"Epoch: {epoch_counter}\tLoss: {avg_loss:.4f}\tAccuracy: {avg_accuracy:.2f}")
            if self.args.earlystop:
                if epoch_counter >= 10 and avg_accuracy > 99.0:  #changed top1[0] to avg_accuracy (for epoch-level accuracy)
                    earlystop_epoch +=1
                else:
                    earlystop_epoch = 0
                if earlystop_epoch >=3:
                    break

            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                # 'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(self.args.output_path, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.output_path}.")

        logging.info("Training has finished.")

        # ckpt = torch.load('checkpoint_0200.pth.tar')
        # self.model.load_state_dict(ckpt['state_dict'])

        with torch.no_grad():
            self.model.eval()
            bs_ = 1024
            out = np.concatenate([self.model(data[0][i:i + bs_].to(self.args.device)).to('cpu').numpy()
                                  for i in range(0, len(data[0]), bs_)], axis=0)
            embeddings_df = pd.DataFrame(out, index=namelist)
            outfile = self.args.output_path + '/embeddings.tsv'
            embeddings_df.to_csv(outfile, sep='\t', header=True)

    def train_addpretrain(self, train_loader, data, namelist):
        """
        Train the SimCLR model with an additional pre-trained k-mer model.

        :param train_loader: Data loader for training.
        :param data: Input data.
        :param namelist: List of sequence names.
        """
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.output_path, self.args)

        earlystop_epoch=0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        # logging.info(f"Training with cpu: {self.args.disable_cuda}.")

        kmer_len = self.args.llm_embedding_dim
        
        print(f'Model: {self.args.model_name}, Embedding Size: {kmer_len}')
        logging.info('kmer_len:\t' + str(kmer_len) + '\n')

        checkpoint = load_checkpoint(self.args)
        if checkpoint is not None:
            resume_epoch = checkpoint['epoch']
            model_state_dict = checkpoint['state_dict']
            optimizer_state_dict = checkpoint['optimizer']

            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            resume_epoch = 0
        
        skipped = 0
        for epoch_counter in range(self.args.epochs):
            if resume_epoch > skipped:
                skipped+=1
                continue

            if epoch_counter == 0:
                epoch_loss1 = 0
                epoch_loss2 = 0
                epoch_loss = 0

            # modified
            for contig_features in tqdm(train_loader):
                contig_features = torch.cat(contig_features, dim=0)

                contig_features = contig_features.to(self.args.device)
                # print(contig_features.shape)

                with autocast(enabled=self.args.fp16_precision):
                    if self.args.addcovloss and not self.args.addkmerloss:
                        if self.args.pretrain_kmer_model_path !='no':
                            features, covemb, kmeremb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                            # print(contig_features[:2, :-kmer_len])
                        else:
                            features, covemb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len]) #passing features (last embedding_dim rows) as x and coverage (first total-embedding_dim rows) as x2
                            # print(contig_features[:2, :-kmer_len])
                        #print(f'[DEBUG] llm_embedding[0]: {contig_features[0, -kmer_len:]}')
                        #print(f'[DEBUG] coverage embedding[0]: {contig_features[0, :-kmer_len]}')

                        logits, labels = self.info_nce_loss(features)
                        loss1 = self.criterion(logits, labels)
                        logits2, labels2 = self.covmodel_info_nce_loss(covemb)
                        loss2 = self.criterion(logits2, labels2)
                        if self.args.lossbalance:
                            if epoch_counter == 0:
                                weight_lambdaloss2 = 1
                            loss = loss1 + weight_lambdaloss2 * loss2
                        else:
                            loss = loss1 + self.args.lambdaloss2 * loss2
                    elif self.args.addcovloss and self.args.addkmerloss:
                        features, covemb, kmeremb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                        # print(len(features[0]))
                        # print(len(covemb[0]))
                        # print(len(kmeremb[0]))
                        logits, labels = self.info_nce_loss(features)
                        loss1 = self.criterion(logits, labels)
                        logits2, labels2 = self.covmodel_info_nce_loss(covemb)
                        loss2 = self.criterion(logits2, labels2)
                        logits3, labels3 = self.kmermodel_info_nce_loss(kmeremb)
                        loss3 = self.criterion(logits3, labels3)
                        loss = loss1 + self.args.lambdaloss2 * loss2   + self.args.lambdakmerloss2 *  loss3
                    else:
                        # features, covemb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                        if self.args.pretrain_kmer_model_path !='no':
                            features, covemb, kmeremb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                            # print(contig_features[:2, :-kmer_len])
                        else:
                            features, covemb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                            # print(contig_features[:2, :-kmer_len])
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if epoch_counter == 0:
                    if self.args.addcovloss:
                        epoch_loss1 += loss1.data.item()
                        epoch_loss2 += loss2.data.item()
                        epoch_loss += loss.data.item()


            top1, top5 = accuracy(logits, labels, topk=(1, 5))

            if not self.args.notuse_scheduler:
                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if self.args.earlystop:
                if epoch_counter >= 10 and top1[0] > 99.0:
                    earlystop_epoch +=1
                else:
                    earlystop_epoch = 0

                if earlystop_epoch >=3:
                    break

            if self.args.addcovloss:
                if epoch_counter == 0:
                    logging.debug(f"Epoch: {epoch_counter}\tcoverage model loss: {loss2}")
                    if self.args.lossbalance:
                        weight_lambdaloss2 = epoch_loss1 / epoch_loss2
                        logging.debug(f"Epoch: {epoch_counter}\tweight_lambdaloss2: {weight_lambdaloss2}")

            if self.args.addkmerloss:
                if epoch_counter == 0:
                    logging.debug(f"Epoch: {epoch_counter}\tkmer model loss: {loss3}")

            save_checkpoint({
                'epoch': epoch_counter,
                # 'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.args.output_path, 'checkpoint.pt'))
            
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            # 'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.args.output_path, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.args.output_path}.")

        # ckpt = torch.load('/checkpoint_0200.pth.tar')

        with torch.no_grad():
            self.model.eval()
            bs_ = 1024

            print(len(data))

            out = np.concatenate([self.model(data[0][i:i + bs_, -kmer_len:].to(self.args.device),
                                             data[0][i:i + bs_, :-kmer_len].to(self.args.device))[0].to('cpu').numpy()
                                  for i in range(0, len(data[0]), bs_)], axis=0)
            embeddings_df = pd.DataFrame(out, index=namelist)

            outfile = self.args.output_path + '/embeddings.tsv'
            embeddings_df.to_csv(outfile, sep='\t', header=True)


            covout = np.concatenate([self.model(data[0][i:i + bs_, -kmer_len:].to(self.args.device),
                                                data[0][i:i + bs_, :-kmer_len].to(self.args.device))[1].to('cpu').numpy()
                                     for i in range(0, len(data[0]), bs_)], axis=0)
            embeddings_df = pd.DataFrame(covout, index=namelist)
            outfile = self.args.output_path + '/covembeddings.tsv'
            embeddings_df.to_csv(outfile, sep='\t', header=True)

    def covmodeltrain(self, train_loader):
        """
        Train the coverage model.

        :param train_loader: Data loader for training.
        """
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.output_path, self.args)

        earlystop_epoch = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        # logging.info(f"Training with cpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.covmodelepochs):
            # modified
            for contig_features in tqdm(train_loader):
                contig_features = torch.cat(contig_features, dim=0)

                contig_features = contig_features.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(contig_features[:, :-self.args.llm_embedding_dim])
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()


            top1, top5 = accuracy(logits, labels, topk=(1, 5))

            if not self.args.notuse_scheduler:
                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if self.args.earlystop:
                if epoch_counter >= 10 and top1[0] > 99.0:
                    earlystop_epoch +=1
                else:
                    earlystop_epoch = 0
                if earlystop_epoch >=3:
                    break

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.covmodelepochs,
            # 'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.args.output_path, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.args.output_path}.")

        # ckpt = torch.load('checkpoint_0200.pth.tar')
        # self.model.load_state_dict(ckpt['state_dict'])

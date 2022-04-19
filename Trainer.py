import torch.optim as optim
from torch.nn import Dropout
from torch.autograd import Variable
from data_tools import post_solver, inverse_temp_to_num
import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, model, loss=None, weight=None, vocab_dict=None, vocab_list=None, data_loader=None, batch_size=32, decode_classes_dict=None, decode_classes_list=None,
                 cuda_use=True, print_every=10, checkpoint_dir_name=None):
        self.model = model
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.data_loader = data_loader
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.cuda_use = cuda_use
        self.print_every = print_every
        self.optimizer = optim.Adam(model.parameters())
        self.batch_size = batch_size
        if loss is None:
            self.criterion = nn.NLLLoss(weight=weight, reduction='mean')
        else:
            self.criterion = loss

    def train(self, model, epoch_num=100, start_epoch=0, valid_every=10):
        train_list = self.data_loader.train_data
        valid_list = self.data_loader.valid_data
        best_valid = 0
        path = ""

        for epoch in range(start_epoch, epoch_num):
            model.encoder_optimizer.step()
            model.prediction_optimizer.step()
            model.generation_optimizer.step()
            model.merge_optimizer.step()

            start_step = 0
            total_num = 0
            total_loss = 0
            total_acc_num = 0
            model.train()
            print("Epoch " + str(epoch+1) + " start training!")
            for batch in self.data_loader.yield_batch(train_list, self.batch_size):
                input = batch['batch_encode_pad_idx']
                input_len = batch['batch_encode_len']
                target = batch['batch_decode_pad_idx']
                target_len = batch['batch_decode_len']
                function_ans = batch['batch_ans']
                num_list = batch['batch_num_list']
                batch_num_count = batch['batch_num_count']
                batch_num_index_list = batch['batch_num_index_list']
                nums_stack_batch = batch['nums_stack_batch']

                model.prediction.train()
                model.encoder.train()
                model.generation.train()
                model.merge.train()

                batch_size = len(input)
                total_num += batch_size

                input = Variable(torch.LongTensor(input))
                target = Variable(torch.LongTensor(target))

                input = input.transpose(0, 1)
                target = target.transpose(0, 1)

                if self.cuda_use:
                    input = input.cuda()
                    target = target.cuda()

                model.encoder_optimizer.zero_grad()
                model.prediction_optimizer.zero_grad()
                model.generation_optimizer.zero_grad()
                model.merge_optimizer.zero_grad()

                loss = model(input, input_len, target, target_len, batch_num_count, self.data_loader.generate_op_index, batch_num_index_list, nums_stack_batch)
                total_loss += loss

                model.encoder_scheduler.step()
                model.prediction_scheduler.step()
                model.generation_scheduler.step()
                model.merge_scheduler.step()

            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # CLIP=1
            #     self.optimizer.step()
            #
            #     # train_ans_acc = self.evaluate(model, train_list)
            #
            #     start_step += 1
            #     if start_step % self.print_every == 0:
                print("Step %d Batch Loss: %.5f  |  Epoch %d Batch Train Loss: %.2f" % (start_step, total_loss/total_num, epoch+1, loss))

            # if (epoch+1) % valid_every == 0 and epoch > 0:
            #     valid_ans_acc = self.evaluate(model, valid_list)
            #     if valid_ans_acc > best_valid:
            #         best_valid = valid_ans_acc
            #         path = os.path.join('./model/', "epoch_"+str(epoch+1)+"_result"+str(100*best_valid/len(valid_list))+".pt")
            #         torch.save(model.state_dict(), path)
            #     print("Epoch %d Batch Valid Acc: %.2f  Acc: %d / %d" % (epoch+1, 100*valid_ans_acc/len(valid_list), valid_ans_acc, len(valid_list)))
        #
        #     print("Epoch %d Batch Train Acc: %.2f  Acc: %d / %d" % (epoch + 1, total_acc_num / len(train_list)*100, total_acc_num, len(train_list)))
        # print("Epoch %d Best Valid Acc: %.2f" % (epoch_num, 100*best_valid/len(valid_list)))
        return path

    def evaluate(self, model, data):
        model.eval()
        epoch_loss = 0
        total_acc_num = 0
        for batch in self.data_loader.yield_batch(data, self.batch_size):
            input = batch['batch_encode_pad_idx']
            input_len = batch['batch_encode_len']
            target = batch['batch_decode_pad_idx']
            target_len = batch['batch_decode_len']
            function_ans = batch['batch_ans']
            num_list = batch['batch_num_list']

            batch_size = len(input)

            input = Variable(torch.LongTensor(input))
            target = Variable(torch.LongTensor(target))

            input = input.transpose(0, 1)
            target = target.transpose(0, 1)

            if self.cuda_use:
                input = input.cuda()
                target = target.cuda()

            output = model(input, target, input_len, target_len)

            classes_len = output.shape[-1]
            output = output[1:].view(-1, classes_len)
            target = target[1:].contiguous().view(-1)
            if self.cuda_use:
                output = output.cuda()
                target = target.cuda()
            total_acc_num += self.get_ans_acc(output, function_ans, batch_size, num_list)

            loss = self.criterion(output, target)
            epoch_loss += loss

            # symbol_list = torch.cat([i.topk(1)[1] for i in output], 0)
            # non_padding = target.ne(self.TRG_PAD_IDX)
            # correct = symbol_list.eq(target).masked_select(non_padding).sum().item()  # data[0]
            # match += correct
            # total += non_padding.sum().item()

        return total_acc_num

    def get_ans_acc(self, output, function_ans, batch_size, num_list):
        acc = 0
        output = output.view(-1, batch_size, len(self.decode_classes_list))
        output = output.transpose(0, 1)
        for i in range(len(output)):
            templates = self.get_template(output[i])
            # print(templates)
            try:
                equ = inverse_temp_to_num(templates, num_list[i])
                # print(equ)
                predict_ans = post_solver(equ)
                if abs(float(predict_ans) - float(function_ans[i])) < 1e-5:
                    acc += 1
            except:
                acc += 0
        return acc

    def get_template(self, pred):
        templates = []
        for vec in pred:
            idx = vec.argmax(0).item()
            if idx == self.decode_classes_dict['PAD_token'] or idx == self.decode_classes_dict['END_token']:
                break
            templates.append(self.decode_classes_list[idx])
        return templates








import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import torch.nn as nn
from utils.data_utils import read_client_data
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.optimizers.fedoptimizer import pFedMeOptimizer_edge
from utils.dlg import DLG

class Edge_Server(object):
    def __init__(self, args, **kwargs):
        # Set up the main attributes
        self.args = args
        self.edge_model_1 = copy.deepcopy(args.edge_model)   #定义edge_model backbone
        self.edge_model_2 = copy.deepcopy(args.edge_model)
        self.device = args.device
        #self.id = id
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        #self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.edge_learning_rate = args.edge_learning_rate

        self.has_BatchNorm = False
        for layer in self.edge_model_1.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        for layer in self.edge_model_2.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        #self.edge2_model = copy.deepcopy(args.model)

        

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.loss = nn.CrossEntropyLoss()

        self.edge_rounds = args.edge_rounds

        
        
        #记录客户端上传的模型参数
        self.clients_1 = []   #客户端列表
        self.clients_2 = []
        self.clients = []
        #self.clients = self.clients_1 + self.clients_2
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.uploaded_weights_2 = []
        self.uploaded_ids_2 = []
        self.uploaded_models_2 = []

        #self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.edge_rs_test_acc = []
        self.edge_rs_test_auc = []
        self.edge_rs_train_loss = []

        self.edge_rs_test_acc_2= []
        self.edge_rs_test_auc_2 = []
        self.edge_rs_train_loss_2 = []
    

    def set_clients(self, clientObj):
        #遍历所有客户端，创建客户端对象
        for i, train_slow, send_slow in zip(range(50), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            #创建客户端对象
            client = clientObj(self.args,    #clientObj:客户端对象的类型，用于实例化客户端
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients_1.append(client)

        for i, train_slow, send_slow in zip(range(50,100), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            #创建客户端对象
            client = clientObj(self.args,    #clientObj:客户端对象的类型，用于实例化客户端
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients_2.append(client)
        #print("Clients:", self.clients)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models1(self):  #将边缘模型分组下发到客户端
        assert (len(self.clients_1) > 0)

        for client in self.clients_1:
            start_time = time.time()
            #分组下发边缘服务器
            client.set_parameters(self.edge_model_1) #把边缘模型深拷贝给本地模型

    def send_models2(self):  #将边缘模型分组下发到客户端
        assert (len(self.clients_2) > 0)

        for client in self.clients_2:
            start_time = time.time()
            #分组下发边缘服务器
            client.set_parameters(self.edge_model_2) #把边缘模型深拷贝给本地模型
    


            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def receive_models(self):  #接收客户端上传的模型参数
        assert (len(self.clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []   # 用于存储客户端上传的模型参数
        tot_samples = 0
        for client in self.clients_1:
            
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)   #client.id 客户端编号
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)  #添加客户端上传的模型到uploaded_models
        for i, w in enumerate(self.uploaded_weights):  
            self.uploaded_weights[i] = w / tot_samples   #聚合权重

    def receive_models_2(self):  #接收客户端上传的模型参数
        assert (len(self.clients) > 0)

        self.uploaded_ids_2 = []
        self.uploaded_weights_2  = []
        self.uploaded_models_2  = []   # 用于存储客户端上传的模型参数
        tot_samples = 0
        for client in self.clients_2:
            
            tot_samples += client.train_samples
            self.uploaded_ids_2 .append(client.id)   
            self.uploaded_weights_2 .append(client.train_samples)
            self.uploaded_models_2 .append(client.model)  #添加客户端上传的模型到uploaded_models
        for i, w in enumerate(self.uploaded_weights_2 ):  
            self.uploaded_weights_2[i] = w / tot_samples   #聚合权重

    
    def aggregate_parameters1(self):
        assert (len(self.uploaded_models) > 0)

        self.edge_model_1 = copy.deepcopy(self.uploaded_models[0])  
        for param in self.edge_model_1.parameters():
            param.data.zero_() #初始边缘模型置零
           
        for w, client_model,client_id in zip(self.uploaded_weights, self.uploaded_models, self.uploaded_ids):
            #print("uploaded_models:",type(self.uploaded_models),self.uploaded_models,"uploaded_ids:",type(self.uploaded_ids),self.uploaded_ids)
            for edge_server_param,client_param in zip(self.edge_model_1.parameters(),client_model.parameters()):
                edge_server_param.data += client_param.data.clone() * w

    def aggregate_parameters2(self):
        assert (len(self.uploaded_models_2) > 0)

        self.edge_model_2 = copy.deepcopy(self.uploaded_models_2[0])  
        for param in self.edge_model_2.parameters():
            param.data.zero_() #初始边缘模型置零
           
        for w, client_model,client_id in zip(self.uploaded_weights_2 , self.uploaded_models_2 , self.uploaded_ids_2):
            #print("uploaded_models:",type(self.uploaded_models),self.uploaded_models,"uploaded_ids:",type(self.uploaded_ids),self.uploaded_ids)
            for edge_server_param,client_param in zip(self.edge_model_2.parameters(),client_model.parameters()):
                edge_server_param.data += client_param.data.clone() * w

    def set_parameters(self, edge_model):
        for new_param, old_param_1, old_param_2 in zip(edge_model.parameters(), self.edge_model_1.parameters(), self.edge_model_2.parameters()):
            old_param_1.data = new_param.data.clone()
            old_param_2.data = new_param.data.clone()

    def clone_model(self, edge_model, target):
        for param, target_param in zip(edge_model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, edge_model, new_params):
        for param, new_param in zip(edge_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
        #     algo1 = algo + "_" + self.goal + "_" + str(self.times)
        #     with h5py.File(result_path + "{}.h5".format(algo1), 'w') as hf:
        #         hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
        #         hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
        #         hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        if (len(self.rs_test_acc_per)):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
    
    
    def edge_test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def edge_train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def edge_evaluate(self, acc=None, loss=None):
        stats = self.edge_test_metrics()
        stats_train = self.edge_train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        if acc == None:
            self.edge_rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        if loss == None:
            self.edge_rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        
        print("\n Intermediate Averaged Train Loss: {:.4f}".format(train_loss))
        print(" Intermediate Averaged Test Accurancy: {:.4f}".format(test_acc))
        print(" Intermediate Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

          
class Edge_pFedMe(Edge_Server):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)
        self.clients = self.clients_1 + self.clients_2

        self.lamda1 = args.lamda1

        self.edge_learning_rate = args.edge_learning_rate

        self.edge_local_params_1 = copy.deepcopy(list(self.edge_model_1.parameters()))
        self.edge_personalized_params_1 = copy.deepcopy(list(self.edge_model_1.parameters()))
        self.edge_local_params_2 = copy.deepcopy(list(self.edge_model_2.parameters()))
        self.edge_personalized_params_2 = copy.deepcopy(list(self.edge_model_2.parameters()))

        self.edge_rs_train_acc_per = []
        self.edge_rs_train_loss_per = []
        self.edge_rs_test_acc_per = []

        self.edge_rs_train_acc_per_2 = []
        self.edge_rs_train_loss_per_2 = []
        self.edge_rs_test_acc_per_2 = []


        self.Budget = []

    def train(self):
        for i in range(self.edge_rounds+1):
            s_t = time.time()
            #self.selected_clients = self.select_clients()

            self.previous_edge_model_1 = copy.deepcopy(list(self.edge_model_1.parameters()))
            self.previous_edge_model_2 = copy.deepcopy(list(self.edge_model_2.parameters()))

            #if i >= 1:
            self.send_models1()
            self.send_models2()

            if i%self.eval_gap == 0:
                print(f"\n-------------Edge Round number: {i}-------------")
                print("\nEdge Evaluate edge model")
                self.edge_evaluate()

            for client in self.clients:
                client.train()
            
            #if i%self.eval_gap == 0:
                #print(f"\n-------------Edge Round number: {i}-------------")
                #print("\nEdge Evaluate personaized model")
                #self.edge_evaluate_personalized()
      
            self.receive_models()
            self.aggregate_parameters1()   #聚合客户端更新到edge_model
            self.receive_models_2()
            self.aggregate_parameters2()

            self.edge_personalized_params_1 = copy.deepcopy(list(self.edge_model_1.parameters()))   #聚合客户端更新到edge_personalized_params
            self.edge_personalized_params_2 = copy.deepcopy(list(self.edge_model_2.parameters()))
            
            #self.edge_model_1.train()
            #self.edge_model_2.train()

            #更新边缘本地模型
            for new_param, edge_localweight,previous_edge_localweight in zip(self.edge_personalized_params_1,self.edge_local_params_1,self.previous_edge_model_1):
                edge_localweight = edge_localweight.to(self.device)
                edge_localweight.data = new_param - self.lamda1 * self.edge_learning_rate *(edge_localweight.data - previous_edge_localweight.data)

            self.update_parameters(self.edge_model_1, self.edge_local_params_1)


            for new_param, edge_localweight,previous_edge_localweight in zip(self.edge_personalized_params_2,self.edge_local_params_2,self.previous_edge_model_2):
                edge_localweight = edge_localweight.to(self.device)
                edge_localweight.data = new_param - self.lamda1 * self.edge_learning_rate *(edge_localweight.data - previous_edge_localweight.data)
            
            self.update_parameters(self.edge_model_2, self.edge_local_params_2)

            

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        #if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
            #break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nEdge Best accuracy.")
        print(max(self.edge_rs_test_acc))
        print("\nAverage time cost per edge round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    
    def set_parameters(self, edge_model):
        for new_param, old_param_1, old_param_2, edge_local_param_1,edge_local_param_2 in zip(edge_model.parameters(), self.edge_model_1.parameters(), self.edge_model_2.parameters(), self.edge_local_params_1,self.edge_local_params_2):
            old_param_1.data = new_param.data.clone()
            old_param_2.data = new_param.data.clone()
            edge_local_param_1.data = new_param.data.clone()
            edge_local_param_2.data = new_param.data.clone()

      
    def edge_test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct

    def edge_train_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def edge_evaluate_personalized(self):
        #self.update_parameters(self.edge_model_1, self.edge_personalized_params_1)
        #self.update_parameters(self.edge_model_2, self.edge_personalized_params_2)
        stats = self.edge_test_metrics_personalized()
        stats_train = self.edge_train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.edge_rs_test_acc_per.append(test_acc)
        self.edge_rs_train_acc_per.append(train_acc)
        self.edge_rs_train_loss_per.append(train_loss)

        self.print_(test_acc, train_acc, train_loss)

    def edge_test_metrics_personalized_2(self):
        num_samples = []
        tot_correct = []
        for c in self.clients_2:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients_2]
        return ids, num_samples, tot_correct

    def edge_train_metrics_personalized_2(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients_2:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def edge_evaluate_personalized_2(self):
        #self.update_parameters(self.edge_model_1, self.edge_personalized_params_1)
        #self.update_parameters(self.edge_model_2, self.edge_personalized_params_2)
        stats = self.edge_test_metrics_personalized_2()
        stats_train = self.edge_train_metrics_personalized_2()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.edge_rs_test_acc_per_2.append(test_acc)
        self.edge_rs_train_acc_per_2.append(train_acc)
        self.edge_rs_train_loss_per_2.append(train_loss)

        self.print_(test_acc, train_acc, train_loss)


    

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
        #     algo1 = algo + "_" + self.goal + "_" + str(self.times)
        #     with h5py.File(result_path + "{}.h5".format(algo1), 'w') as hf:
        #         hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
        #         hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
        #         hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        if (len(self.rs_test_acc_per)):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)






  
 
        

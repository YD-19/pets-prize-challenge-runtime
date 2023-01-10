import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

import torch.utils.data.dataset as Dataset
import torch.utils.data as data
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle


import math
def convertToNumber (s):
    if pd.isna(s):
        return int(0)
    return float(int.from_bytes(s.encode(), 'little'))

def convertFromNumber (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()


class SwiftModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst


def add_finalreceiver_col(swift_data: pd.DataFrame):
    """Adds column identifying FinalReciver to SWIFT dataset inplace. Required for
    joining to the bank data.

    See https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/589/#end-to-end-transactions
    """
    uetr_groups_train = swift_data.sort_values("Timestamp").groupby("UETR")
    swift_data["FinalReceiver"] = swift_data["UETR"].map(
        uetr_groups_train.Receiver.last().to_dict()
    )
    return swift_data

# def join_flags_to_swift_data(swift_df: pd.DataFrame, bank_df: pd.DataFrame):
#     """Join BeneficiaryFlags columns onto SWIFT dataset."""
#     # Join beneficiary account flags
#     swift_df = (
#         swift_df.reset_index()
#         .merge(
#             right=bank_df[["Bank", "Account", "Flags"]].rename(
#                 columns={"Flags": "BeneficiaryFlags"}
#             ),
#             how="left",
#             left_on=["FinalReceiver", "BeneficiaryAccount"],
#             right_on=["Bank", "Account"],
#         )
#         .set_index("MessageId")
#     )
#     return swift_df

def join_flags_to_swift_data(swift_df: pd.DataFrame, bank_df: pd.DataFrame):
    """Join BeneficiaryFlags columns onto SWIFT dataset."""
    # Join beneficiary account flags
    swift_df = (
        swift_df.reset_index()
        .merge(
            right=bank_df[["Bank", "Name","Account","Street","CountryCityZip", "Flags"]].rename(
                columns={"Flags": "BeneficiaryFlags"}
            ),
            how="left",
            left_on=["FinalReceiver", "BeneficiaryAccount"],
            right_on=["Bank", "Account"],
        )
        .set_index("MessageId")
    )
    return swift_df



def swiftdataprocessing(swift_df_):
    swift_df = swift_df_.copy()
    feature_name = ["InstructedAmount","SettlementAmount","SettlementCurrency","InstructedCurrency"]
    batch_size = 128
    y=swift_df.pop("Label")
    x=swift_df[feature_name].copy()
    Currency_transfer_table = {'AUD':1, 'EUR':2, 'CAD':3, 'GBP':4,
               'INR':5, 'JPY':6, 'NZD':7, 
               'USD':8}
    x["SettlementCurrency"] = swift_df["SettlementCurrency"].apply(lambda x: Currency_transfer_table.get(x))
    x["InstructedCurrency"] = swift_df["InstructedCurrency"].apply(lambda x: Currency_transfer_table.get(x))
    x=(x-x.mean())/x.std()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=2018)
    train_set=TensorDataset(torch.from_numpy(x_train.values).to(torch.float32),torch.from_numpy(y_train.values).to(torch.float32))
    test_set = TensorDataset(torch.from_numpy(x_val.values).to(torch.float32),torch.from_numpy( y_val.values).to(torch.float32))
    train_loader = DataLoader( train_set,batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,drop_last=True)
    return train_loader,test_loader

def Fullfill(x):
    if pd.isna(x):
        return float(0)
    else:
        return float(x)

def bankdataprocessing(swift_df_):
    swift_df = swift_df_.copy()
    feature_name = ["Name","Account","Street","CountryCityZip"]
    batch_size = 128
    y=swift_df.pop("BeneficiaryFlags").apply(lambda x: Fullfill(x))
    x=swift_df[feature_name].copy()

    x["Name"] = swift_df["Name"].apply(lambda x: convertToNumber(x))
    x["Account"] = swift_df["Account"].apply(lambda x: convertToNumber(x))
    x["Street"] = swift_df["Street"].apply(lambda x: convertToNumber(x))
    x["CountryCityZip"] = swift_df["CountryCityZip"].apply(lambda x: convertToNumber(x))
    x=(x-x.mean())/x.std()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=2018)
    train_set=TensorDataset(torch.from_numpy(x_train.values).to(torch.float32),torch.from_numpy(y_train.values).to(torch.float32))
    test_set = TensorDataset(torch.from_numpy(x_val.values).to(torch.float32),torch.from_numpy( y_val.values).to(torch.float32))
    train_loader = DataLoader( train_set,batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,drop_last=True)
    return train_loader,test_loader


class BankModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=pd.NA, strategy="constant", fill_value="-1"
                    ),
                ),
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if len(self.pipeline.named_steps["model"].classes_) == 1:
            return pd.Series([0.0] * X.shape[0], index=X.index)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst


class SwiftNet():
    def __init__(self):
        self.net=nn.Sequential(
            nn.Sequential(nn.Linear(4, 20),nn.Dropout(0.3),nn.Sigmoid()),
            nn.Sequential(nn.Linear(20, 10),nn.Dropout(0.3),nn.Sigmoid()),
            nn.Sequential(nn.Linear(10, 1),nn.Dropout(0.3),nn.Sigmoid())
            )
        self.epochs = 5
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        print(self.device)
        self.train_loss_list=[]
        self.test_loss_list=[]

    def train(self, train_loader):
        self.net.train()
        step = 0
        nums = 0
        train_loss = 0
        for data in train_loader:
            self.optimizer.zero_grad()
            nums+=1
            x, targets = data
            x = x.to(self.device)
            batch_len = len(targets)
            targets = targets.to(self.device)
            y_pred = self.net(x)
            loss = self.loss_fn(y_pred.squeeze(), targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # if (step % 10 == 0):
            #     print("step{},loss:{}".format(step, loss))
            # step += 1
        epoch_loss = train_loss / nums
        self.train_loss_list.append(epoch_loss)
        return epoch_loss

    def predict(self, swift_df_):
        swift_df = swift_df_.copy()
        feature_name = ["InstructedAmount","SettlementAmount","SettlementCurrency","InstructedCurrency"]
        batch_size = 128
        y=swift_df.pop("Label")
        x=swift_df[feature_name].copy()
        Currency_transfer_table = {'AUD':1, 'EUR':2, 'CAD':3, 'GBP':4,
                'INR':5, 'JPY':6, 'NZD':7, 
                'USD':8}
        x["SettlementCurrency"] = swift_df["SettlementCurrency"].apply(lambda x: Currency_transfer_table.get(x))
        x["InstructedCurrency"] = swift_df["InstructedCurrency"].apply(lambda x: Currency_transfer_table.get(x))
        x=(x-x.mean())/x.std()
        test_set = TensorDataset(torch.from_numpy(x.values).to(torch.float32),torch.from_numpy( y.values).to(torch.float32))
        test_loader = DataLoader(test_set, batch_size=batch_size,drop_last=False)
        self.net.eval()
        test_loss = 0
        len_test=0
        nums=0
        y_total=[]
        for data in test_loader:
            nums+=1
            X, targets = data
            X = X.to(self.device)
            targets = targets.to(self.device)
            y_pred = self.net(X)
            len_test+=len(targets)
            y_hat = torch.argmax(y_pred, 1)
            loss = self.loss_fn(y_pred.squeeze(), targets)
            test_loss += loss.item()
            pred_np = y_pred.cpu().detach().numpy()
            for elem in pred_np:
                y_total.append(elem)
        self.test_loss_list.append(test_loss/nums)
        return pd.Series(y_total, index=swift_df_.index)

    def fit(self,train_date):
        for epoch in range(self.epochs):
            # print("{}epoch".format(epoch + 1))
            train_loss=self.train(train_date)
        # loss= self.predict(self, test_loader)
            print("{}epoch,train error{}".format(epoch + 1,train_loss))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        print("Swift Model Saved!")
        # joblib.dump(self.pipeline, path)

    # @classmethod
    def load(self,path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        print("Swift Model Loaded!")
        # return self.net

class BankNet():
    def __init__(self):
        self.net=nn.Sequential(
            nn.Sequential(nn.Linear(4, 20),nn.Dropout(0.3),nn.Sigmoid()),
            nn.Sequential(nn.Linear(20, 10),nn.Dropout(0.3),nn.Sigmoid()),
            nn.Sequential(nn.Linear(10, 1),nn.Dropout(0.3),nn.Sigmoid())
            )
        self.epochs = 5
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        print(self.device)
        self.train_loss_list=[]
        self.test_loss_list=[]

    def train(self, train_loader):
        self.net.train()
        step = 0
        nums = 0
        train_loss = 0
        for data in train_loader:
            self.optimizer.zero_grad()
            nums+=1
            x, targets = data
            x = x.to(self.device)
            batch_len = len(targets)
            targets = targets.to(self.device)
            y_pred = self.net(x)
            loss = self.loss_fn(y_pred.squeeze(), targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # if (step % 10 == 0):
            #     print("step{},loss:{}".format(step, loss))
            # step += 1
        epoch_loss = train_loss /nums
        self.train_loss_list.append(epoch_loss)
        return epoch_loss

    def predict(self, swift_df_):
        swift_df = swift_df_.copy()
        feature_name = ["Name","Account","Street","CountryCityZip"]
        batch_size = 128
        y=swift_df.pop("BeneficiaryFlags").apply(lambda x: Fullfill(x))
        x=swift_df[feature_name].copy()

        x["Name"] = swift_df["Name"].apply(lambda x: convertToNumber(x))
        x["Account"] = swift_df["Account"].apply(lambda x: convertToNumber(x))
        x["Street"] = swift_df["Street"].apply(lambda x: convertToNumber(x))
        x["CountryCityZip"] = swift_df["CountryCityZip"].apply(lambda x: convertToNumber(x))
        x=(x-x.mean())/x.std()
        test_set = TensorDataset(torch.from_numpy(x.values).to(torch.float32),torch.from_numpy( y.values).to(torch.float32))
        test_loader = DataLoader(test_set, batch_size=batch_size,drop_last=False)
        self.net.eval()
        test_loss = 0
        len_test=0
        nums=0
        y_total=[]
        for data in test_loader:
            nums+=1
            X, targets = data
            X = X.to(self.device)
            targets = targets.to(self.device)
            y_pred = self.net(X)
            len_test+=len(targets)
            y_hat = torch.argmax(y_pred, 1)
            loss = self.loss_fn(y_pred.squeeze(), targets)
            test_loss += loss.item()
            pred_np = y_pred.cpu().detach().numpy()
            for elem in pred_np:
                y_total.append(elem)
        self.test_loss_list.append(test_loss/nums)
        return pd.Series(y_total, index=swift_df_.index)

    def fit(self, train_date):
        for epoch in range(self.epochs):
            # print("{}epoch".format(epoch + 1))
            train_loss=self.train(train_date)
        # loss= self.predict(self, test_loader)
            print("{}epoch,train error{}".format(epoch + 1,train_loss))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        print("Bank Model Saved!")
        # joblib.dump(self.pipeline, path)

    # @classmethod
    def load(self,path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        print("Bank Model Loaded!")
        # return self.net


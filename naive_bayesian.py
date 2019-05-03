# -*- coding: utf-8 -*-
import csv
import random
import numpy
import math
import time
import sys

def main(argv):
    filename = "data.csv"
    ratio = 0.8
    repeats = 1
    result_set = []
    for i in range(repeats):
        try:
            start = time.time()
            # 读取数据
            records = read_csv(filename)
            # 划分训练集和测试集
            (trainset, testset) = split_dataset(records, ratio)
            # 将训练集里的记录按不同职位分割
            partitions = partition_trainset(trainset)
            # 统计每个职位每个属性取不同值的数量
            counter = count_attributes(partitions)
            # 计算每个职位每个属性取不同值的概率
            probabilities = calculate_probabilities(partitions, counter)
            # 对测试集中的记录进行预测
            predict_testset(testset, probabilities, result_set)
            end = time.time()
            print("Cost: {0} s".format(end - start))
            print("**********")

            result = calculate_probabilities(partition_trainset(records), count_attributes(partition_trainset(records)))
            with open("result.txt", "w") as result_file:
                for item in result:
                    result_file.write(item)
                    result_file.write("\n")
                    for attr in result[item]:
                        for value in attr:
                            result_file.write("({0}, {1})".format(value, attr[value]))
                        result_file.write("\n")
        except (IOError, TypeError, ValueError) as identifier:
            print(identifier)
    print(numpy.mean(result_set))

def read_csv(filename):
    records = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for record in reader:
            records.append(record)
        print("**********")
        print("Data Loaded Successfully!")
        print("{0}: {1} rows".format(filename, len(records)))
        print("**********")
    return records

def split_dataset(dataset, ratio):
    trainset_capacity = int(len(dataset) * ratio)
    trainset = []
    testset = list(dataset)
    count = 0
    while count < trainset_capacity:
        index = random.randrange(0, len(testset))
        trainset.append(testset.pop(index))
        count += 1
    print("Split the Dataset Successfully!")
    print("Train Set: {0} rows".format(len(trainset)))
    print("Test Set: {0} rows".format(len(testset)))
    print("**********")
    return (trainset, testset)

def partition_trainset(trainset):
    partitions = {}
    for record in trainset:
        if record[0] not in partitions:
            partitions[record[0]] = []
        partitions[record[0]].append(record)
    print("Partition the Train Set Successfully!")
    for key in partitions:
        print("{0}: {1} rows".format(key, len(partitions[key])))
    print("**********")
    return partitions

def count_attributes(partitions):
    counter = {}
    for key in partitions:
        for index in range(1, 19):
            attr_map = {}
            for item in partitions[key]:   
                if item[index] not in attr_map:
                    attr_map[item[index]] = 0
                attr_map[item[index]] += 1
            if key not in counter:
                counter[key] = []
            counter[key].append(attr_map)
    return counter

def calculate_probabilities(partitions, counter):
    probabilities = {}
    for key in counter:
        length = len(partitions[key])
        for dictionary in counter[key]:
            attr_map = {}
            for item in dictionary:
                    attr_map[item] = float(dictionary[item]) / length
            if key not in probabilities:
                probabilities[key] = []
            probabilities[key].append(attr_map)
    print("Calculate Probabilities Successfully!")
    print("**********")
    return probabilities
                
def predict_testset(testset, probabilities, result_set):
    actual_values = []
    predicted_values = []
    count = 0
    for record in testset:
        actual_values.append(record[0])
    for record in testset:
        value_dict = {}
        # product
        product_probability = calculate_prediction("product", probabilities, record)
        value_dict[product_probability] = "product"
        # technical
        technical_probability = calculate_prediction("technical", probabilities, record)
        value_dict[technical_probability] = "technical"
        # sales
        sales_probability = calculate_prediction("sales", probabilities, record)
        value_dict[sales_probability] = "sales"
        # operation
        operation_probability = calculate_prediction("operation", probabilities, record)
        value_dict[operation_probability] = "operation"
        # finance
        finance_probability = calculate_prediction("finance", probabilities, record)
        value_dict[finance_probability] = "finance"
        # HR
        HR_probability = calculate_prediction("HR", probabilities, record)
        value_dict[HR_probability] = "HR"
        predicted_values.append(value_dict[max(product_probability, technical_probability, sales_probability, operation_probability, finance_probability, HR_probability)])
    for i in range(len(actual_values)):
        if actual_values[i] == predicted_values[i]:
            count += 1
    result_set.append(count / len(actual_values))
    print("Accuracy: {0}".format(count / len(actual_values)))
    print("**********")

def calculate_prediction(item, probabilities, record):
    probability = 1
    count = 0
    for i in set(range(1,19)):
        if record[i] not in probabilities[item][count]:
            probability *= 0
        else:
            probability *= probabilities[item][count][record[i]]
        count += 1
    return probability

if __name__ == '__main__':
    main(sys.argv)

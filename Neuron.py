import time
import random
import math

class Neuron:
    def __init__(self,  bias: float, weights : list) -> None:
        self.bias = bias
        self.threshold = -1 * bias
        self.weights = weights
        self.output = 0

    def set_bias(self, bias: float) -> None:
        self.bias = bias
        self.threshold = -1 * bias    

    def __int__(self) -> int:
        return int(self.output)
    def __str__(self) -> str:
        return str(self.output)
    def __float__(self) -> float:
        return float(self.output)
    
class Perceptron(Neuron):
    def __init__(self, bias: float, weights: list) -> None:
        super().__init__(bias, weights)

    
    def activate(self, inputs: list) -> float:
        output = sum([inputs[i] * self.weights[i] for i in range(len(inputs))])
        if output > self.threshold:
            self.output = 1
            return 1
        self.output = 0
        return 0

class Sigmoid(Neuron):
    def __init__(self, bias: float, weights: list) -> None:
        super().__init__(bias, weights)

    def __sigmoid(self, x: float) -> float:
        e = 2.718281828459045
        return 1 / (1 + e ** -x)

    def activate(self, inputs: list) -> float:
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        output += self.bias
        output = self.__sigmoid(output)
        self.output = output

def bitwise_addition(x1: int, x2: int, Nand: Neuron) -> None:
    nand_1 = Nand.activate([x1, x2])
    nand_2_1 = Nand.activate([x1, nand_1])
    nand_2_2 = Nand.activate([nand_1, x2])
    nand_2_3 = Nand.activate([nand_1, nand_1])
    nand_3_1 = Nand.activate([nand_2_1, nand_2_2])
    result = nand_3_1
    carry_bit = nand_2_3
    return result, carry_bit


Nand = Perceptron(3, [-2, -2])
x0 = 1
x1 = 1
print(Nand.activate([x0, x1]))
print(bitwise_addition(x0, x1, Nand))

####







        




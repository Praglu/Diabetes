import math
import numpy as np
import random
import pandas as pd  # CSV files
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('husl')

diabetes = pd.read_csv('diabetes_data_upload.csv')

diabetes.describe()

diabetes.head()

sns.set_theme(style="white")

# Load the example mpg dataset
mpg = sns.load_dataset("mpg")

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="class", y="Age", hue="Gender", size="Age",
            sizes=(15, 90), alpha=.5, palette="muted",
            height=6, data=diabetes)

diabetes.Gender = diabetes.Gender.map({'Female': 0, 'Male': 1})
diabetes['Gender'] = pd.to_numeric(diabetes['Gender'])
diabetes.Polyuria = diabetes.Polyuria.map({'No': 0, 'Yes': 1})
diabetes['Polyuria'] = pd.to_numeric(diabetes['Polyuria'])
diabetes.Polydipsia = diabetes.Polydipsia.map({'No': 0, 'Yes': 1})
diabetes['Polydipsia'] = pd.to_numeric(diabetes['Polydipsia'])
diabetes['sudden weight loss'] = diabetes['sudden weight loss'].map({'No': 0, 'Yes': 1})
diabetes['sudden weight loss'] = pd.to_numeric(diabetes['sudden weight loss'])
diabetes.weakness = diabetes.weakness.map({'No': 0, 'Yes': 1})
diabetes['weakness'] = pd.to_numeric(diabetes['weakness'])
diabetes.Polyphagia = diabetes.Polyphagia.map({'No': 0, 'Yes': 1})
diabetes['Polyphagia'] = pd.to_numeric(diabetes['Polyphagia'])
diabetes['Genital thrush'] = diabetes['Genital thrush'].map({'No': 0, 'Yes': 1})
diabetes['Genital thrush'] = pd.to_numeric(diabetes['Genital thrush'])
diabetes['visual blurring'] = diabetes['visual blurring'].map({'No': 0, 'Yes': 1})
diabetes['visual blurring'] = pd.to_numeric(diabetes['visual blurring'])
diabetes.Itching = diabetes.Itching.map({'No': 0, 'Yes': 1})
diabetes['Itching'] = pd.to_numeric(diabetes['Itching'])
diabetes.Irritability = diabetes.Irritability.map({'No': 0, 'Yes': 1})
diabetes['Irritability'] = pd.to_numeric(diabetes['Irritability'])
diabetes['delayed healing'] = diabetes['delayed healing'].map({'No': 0, 'Yes': 1})
diabetes['delayed healing'] = pd.to_numeric(diabetes['delayed healing'])
diabetes['partial paresis'] = diabetes['partial paresis'].map({'No': 0, 'Yes': 1})
diabetes['partial paresis'] = pd.to_numeric(diabetes['partial paresis'])
diabetes['muscle stiffness'] = diabetes['muscle stiffness'].map({'No': 0, 'Yes': 1})
diabetes['muscle stiffness'] = pd.to_numeric(diabetes['muscle stiffness'])
diabetes.Alopecia = diabetes.Alopecia.map({'No': 0, 'Yes': 1})
diabetes['Alopecia'] = pd.to_numeric(diabetes['Alopecia'])
diabetes.Obesity = diabetes.Obesity.map({'No': 0, 'Yes': 1})
diabetes['Obesity'] = pd.to_numeric(diabetes['Obesity'])
diabetes['class'] = diabetes['class'].map({'Negative': 0, 'Positive': 1})
diabetes['class'] = pd.to_numeric(diabetes['class'])


class DataProcessing:
    # tasowanie zbioru
    @staticmethod
    def shuffle(X):
        for i in range(len(X) - 1, 0, -1):
            j = random.randint(0, i)
            # X[i], X[j] = X[j], X[i] normalnie mo??na tak, ale biblioteka pandas tutaj nie pozwoli
            X.iloc[i], X.iloc[j] = X.iloc[j], X.iloc[i]
        return X

    def splitSet(X):
        # tu podzia?? zbio??w
        split = int(len(X) * 0.8)
        train = X[:split]
        val = X[split:]
        return [train, val]


diabetesMixed = DataProcessing.shuffle(diabetes)
diabetesTrain, diabetesVal = DataProcessing.splitSet(diabetesMixed)

# diabetesTrain
# diabetesVal

diabetesTrain.loc[:, "Age":"Obesity"].to_numpy()

decision = []
for index in diabetesTrain['class']:
    decision.append([index])
print(decision)

# dane wej??ciowe i wyj??ciowe
inputs = diabetesTrain.loc[:, "Age":"Obesity"].to_numpy()
outputs = decision


# stworzenie klasy sieci neuronowej
class NeuralNetwork:

    # stworzenie konstruktora
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        # dla u??atwienia ustawienie wag na 0.50
        self.weights = np.array(
            [[.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50],
             [.50]])
        self.error_history = np.array([])
        self.epoch_list = np.array([])

    # funkcja aktywacyjna ( sigmoidalna ) ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # dane b??d?? przep??ywa??y przez sie?? neuronow?? ( warstwa ukryta )
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # przechodzenie wstecz przez sie?? w celu aktualizacji wag
    def backpropagation(self):
        # self.error  = self.outputs - self.hidden # b????d ??rednio kwadratowy
        self.error = 1 / 2 * (self.outputs - self.hidden) ** 2
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # trenowanie sieci neuronowej przez 40 000 iteracji
    def train(self, epochs=40000):
        for epoch in range(epochs):
            # przep??yw danych i wyprodukowanie danych wyj??ciowych
            self.feed_forward()
            # propagacja wsteczna, czyli przej??cie do ty??u w celu aktualizacji wag
            self.backpropagation()
            # ??ledzenie historii b????d??w w ka??dej epoce
            self.error_history = np.append(self.error_history, np.average(np.abs(self.error)))
            self.epoch_list = np.append(self.epoch_list, epoch)

    # funkcja do przewidywania wynik??w dla nowych i nieznanych danych wej??ciowych                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


# stworzenie sieci
NN = NeuralNetwork(inputs, outputs)

# trenowanie sieci
NN.train()

# skorzystanie z danych validacyjnych
val_input = diabetesVal.loc[:, "Age":"Obesity"].to_numpy()
val_result = diabetesVal.loc[:, "class"].to_numpy()
val_predicted = []

# sprawdzenie predykcji dla wszystkich danych wej??ciowych zbioru walidacyjnego
for row in val_input:
    predict = NN.predict(row)
    val_predicted.append(predict)
# obliczenie skuteczno??ci  
error = val_predicted - val_result
print("Accuracy: ", (error.size - np.count_nonzero(error)) / error.size * 100, "%")

# wy??wietlenie wykresu obrazuj??cego ca??y proces trenowania
plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, np.abs(NN.error_history))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

check_mate = []
while True:
    try:
        agee = int(input("Prosz?? poda?? wiek: "))
        if agee > 0 and agee < 120:
            break
        else:
            print("Prosz?? poda?? odpowiedni wiek!")
    except ValueError:
        print("Prosz?? poda?? liczb??!")
check_mate.append(agee)

while True:
    try:
        genderr = int(input("Prosz?? poda?? p??e??: (1 m????czyzna, 0 kobieta)"))
        if genderr >= 0 and genderr <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(genderr)

while True:
    try:
        polyuriaa = int(input("Czy wyst??puje wielomocz? (1 tak, 0 nie)"))
        if polyuriaa >= 0 and polyuriaa <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(polyuriaa)

while True:
    try:
        polidypsiaa = int(input("Czy wyst??puje wzmo??one pragnienie? (1 tak, 0 nie)"))
        if polidypsiaa >= 0 and polidypsiaa <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(polidypsiaa)

while True:
    try:
        weightt = int(input("Czy wyst??puje spadek wagi? (1 tak, 0 nie)"))
        if weightt >= 0 and weightt <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(weightt)

while True:
    try:
        weak = int(input("Czy wyst??puj?? s??abo??ci? (1 tak, 0 nie)"))
        if weak >= 0 and weak <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(weak)

while True:
    try:
        polifagiaa = int(input("Czy wyst??puje nadmierny apetyt? (1 tak, 0 nie)"))
        if polifagiaa >= 0 and polifagiaa <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(polifagiaa)

while True:
    try:
        gthrush = int(input("Czy wyst??puj?? ple??niawki narz??d??w p??ciowych? (1 tak, 0 nie)"))
        if gthrush >= 0 and gthrush <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(gthrush)

while True:
    try:
        blurr = int(input("Czy wyst??puje rozmycie widzenia? (1 tak, 0 nie)"))
        if blurr >= 0 and blurr <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(blurr)

while True:
    try:
        itch = int(input("Czy wyst??puje sw??dzenie? (1 tak, 0 nie)"))
        if itch >= 0 and itch <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(itch)

while True:
    try:
        irrit = int(input("Czy wyst??puje irytacja? (1 tak, 0 nie)"))
        if irrit >= 0 and irrit <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(irrit)

while True:
    try:
        dheal = int(input("Czy wyst??puje op????nione gojenie? (1 tak, 0 nie)"))
        if dheal >= 0 and dheal <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(dheal)

while True:
    try:
        paresis = int(input("Czy wyst??puje cz????ciowy niedow??ad? (1 tak, 0 nie)"))
        if paresis >= 0 and paresis <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(paresis)

while True:
    try:
        muscle = int(input("Czy wyst??puje sztywno???? mi????ni? (1 tak, 0 nie)"))
        if muscle >= 0 and muscle <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(muscle)

while True:
    try:
        bald = int(input("Czy wyst??puje ??ysienie? (1 tak, 0 nie)"))
        if bald >= 0 and bald <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(bald)

while True:
    try:
        obes = int(input("Czy wyst??puje oty??o????? (1 tak, 0 nie)"))
        if obes >= 0 and obes <= 1:
            break
        else:
            print("Prosz?? poda?? odpowiedni?? odpowied??!")
    except ValueError:
        print("Prosz?? poda?? cyfr??!")
check_mate.append(obes)

print("Szansa na cukrzyc?? wynosi:", NN.predict(check_mate))
